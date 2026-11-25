"""Comprehensive unit and integration test suite for clinaudit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clinaudit import (  # noqa: E402
    CodeDocument,
    TableData,
    __version__,
    audit_cross_table_ci_consistency,
    audit_dca_table,
    audit_directory,
    audit_files,
    audit_linear_coefficients,
    audit_risk_calculator,
    audit_split_leakage,
    audit_stat_file,
    audit_table_stats,
    audit_tabular_split_files,
    audit_xlsx_splits,
    check_dca_curve,
    confusion_matrix_unseen_bounds,
    exact_binomial_test,
    exact_binomial_upper_tail_log10,
    exact_mcnemar_test,
    extract_ci_records,
    extract_numeric_matrix,
    min_mcnemar_p_for_net_diff,
    paired_z_sensitivity_check,
    read_code_file,
    read_delimited,
    read_tables,
    read_xlsx,
    scan_code_document,
    scan_deploy_code_document,
    write_xlsx,
)
from clinaudit.io_utils import col_letters_to_index, index_to_col_letters  # noqa: E402


class TestIOUtils(unittest.TestCase):
    def test_excel_col_conversions(self) -> None:
        self.assertEqual(col_letters_to_index("A"), 0)
        self.assertEqual(col_letters_to_index("Z"), 25)
        self.assertEqual(col_letters_to_index("AA"), 26)
        self.assertEqual(col_letters_to_index("AB"), 27)
        for idx in (0, 1, 25, 26, 27, 51, 52, 701, 702):
            self.assertEqual(col_letters_to_index(index_to_col_letters(idx)), idx)

    def test_xlsx_roundtrip_shared_and_inline_strings(self) -> None:
        sheets = {
            "Summary": [
                ["Model", "AUC", "Valid", "Note"],
                ["XGBoost", 0.872, True, "Primary"],
                ["LR", 0.81, False, None],
            ],
            "Scores": [
                ["ID", "Score", "Extra"],
                [1, 12.5, "ok"],
                [2, None, "sparse"],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p_shared = Path(tmp) / "shared.xlsx"
            p_inline = Path(tmp) / "inline.xlsx"
            write_xlsx(p_shared, sheets, use_inline_strings=False)
            write_xlsx(p_inline, sheets, use_inline_strings=True)

            for p in (p_shared, p_inline):
                loaded = read_xlsx(p)
                self.assertEqual(list(loaded.keys()), ["Summary", "Scores"])
                self.assertEqual(loaded["Summary"][0], ["Model", "AUC", "Valid", "Note"])
                self.assertAlmostEqual(loaded["Summary"][1][1], 0.872)
                self.assertIs(loaded["Summary"][1][2], True)
                self.assertIs(loaded["Summary"][2][2], False)
                self.assertIsNone(loaded["Summary"][2][3])
                self.assertIsNone(loaded["Scores"][2][1])
                self.assertEqual(loaded["Scores"][2][2], "sparse")

    def test_xlsx_rich_text_shared_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "richtext.xlsx"
            write_xlsx(p, {"S1": [["Header", "Val"], ["Row1", 1.5]]})
            # Replace sharedStrings.xml with rich-text <r><t> segments
            buf_files = {}
            with zipfile.ZipFile(p, "r") as zf:
                for name in zf.namelist():
                    buf_files[name] = zf.read(name)
            buf_files["xl/sharedStrings.xml"] = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                b"<si><r><t>Head</t></r><r><t>er</t></r></si>"
                b"<si><t>Val</t></si>"
                b"<si><t>Row1</t></si>"
                b"</sst>"
            )
            with zipfile.ZipFile(p, "w") as zf:
                for name, data in buf_files.items():
                    zf.writestr(name, data)
            loaded = read_xlsx(p)
            self.assertEqual(loaded["S1"][0][0], "Header")

    def test_csv_tsv_and_numeric_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tsv_p = Path(tmp) / "cohort.tsv"
            tsv_p.write_text(
                "patient_id\tage\tegfr\tnotes\toutcome\n"
                "P1\t65\t72.5\tok\t1\n"
                "P2\t58\t84.0\tok\t0\n"
                "P3\t71\t49.2\thigh\t1\n",
                encoding="utf-8",
            )
            rows = read_delimited(tsv_p)
            self.assertEqual(len(rows), 4)
            tbl = read_tables(tsv_p)[0]
            X, y, feats, target = extract_numeric_matrix(tbl)
            self.assertEqual(feats, ["age", "egfr"])
            self.assertEqual(target, "outcome")
            self.assertEqual(X.shape, (3, 2))
            self.assertEqual(y.tolist(), [1.0, 0.0, 1.0])

    def test_ipynb_reader_and_magic_sanitization(self) -> None:
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "markdown", "source": ["# Clinical Pipeline"]},
                {
                    "cell_type": "code",
                    "source": ["%matplotlib inline\n", "import numpy as np\n", "x = 1\n"],
                },
                {
                    "cell_type": "code",
                    "source": ["%%bash\n", "echo hello\n"],
                },
                {
                    "cell_type": "code",
                    "source": ["chads_bin = (cha2ds2_vasc >= 2).astype(int)\n", "auc = roc_auc_score(y, chads_bin)\n"],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            nb_p = Path(tmp) / "analysis.ipynb"
            nb_p.write_text(json.dumps(nb), encoding="utf-8")
            doc = read_code_file(nb_p)
            self.assertEqual(doc.language, "python")
            alerts = scan_code_document(doc)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].rule_id, "EVAL-001")
            self.assertIn("cell 2", alerts[0].location or "")


class TestModule1Leakage(unittest.TestCase):
    def test_clean_continuous_and_low_cardinality_splits(self) -> None:
        rng = np.random.default_rng(101)
        X = rng.normal(size=(300, 8))
        y = rng.binomial(1, 0.3, size=300)
        alerts, diag = audit_split_leakage(X[:220], X[220:], y[:220], y[220:])
        self.assertEqual(alerts, [])
        self.assertEqual(diag["n_test_leaked"], 0)

        # Low-cardinality binary comorbidity table: natural collisions must NOT trigger FP
        X_bin = rng.binomial(1, 0.5, size=(400, 5)).astype(np.float64)
        y_bin = rng.binomial(1, 0.35, size=400).astype(np.float64)
        alerts_bin, diag_bin = audit_split_leakage(X_bin[:300], X_bin[300:], y_bin[:300], y_bin[300:])
        self.assertTrue(diag_bin["cardinality"]["is_low_cardinality"])
        self.assertEqual(alerts_bin, [])

    def test_near_exact_tolerance_and_nan_matching(self) -> None:
        rng = np.random.default_rng(102)
        X_tr = rng.normal(size=(150, 6))
        X_te = rng.normal(size=(60, 6))
        X_tr[0, 2] = np.nan
        # Copy 10 rows with perturbation inside tol=1e-10
        X_te[:10] = X_tr[:10] + rng.uniform(-4e-11, 4e-11, size=(10, 6))
        X_te[0, 2] = np.nan  # Matching NaN position
        alerts, diag = audit_split_leakage(X_tr, X_te, tol=1e-10)
        self.assertTrue(any(a.rule_id == "LEAK-001" for a in alerts))
        self.assertEqual(diag["n_test_leaked"], 10)

        # Perturbation outside tol=1e-10 (e.g., 1e-6) should NOT match
        X_te_far = rng.normal(size=(60, 6))
        X_te_far[:10] = X_tr[:10] + 1e-6
        alerts_far, diag_far = audit_split_leakage(X_tr, X_te_far, tol=1e-10)
        self.assertEqual(diag_far["n_test_leaked"], 0)
        self.assertEqual(alerts_far, [])

    def test_mirror_clone_parity_and_exact_binomial_log10(self) -> None:
        # Exact binomial log10 without float64 underflow
        p_val, log10_p = exact_binomial_upper_tail_log10(100, 100, 0.3)
        self.assertAlmostEqual(log10_p, 100.0 * math.log10(0.3), places=6)
        self.assertAlmostEqual(p_val, 0.3**100, places=60)

        # Extreme underflow beyond 1e-308
        p_ext, log10_ext = exact_binomial_upper_tail_log10(1000, 1000, 0.3)
        self.assertEqual(p_ext, 0.0)
        self.assertAlmostEqual(log10_ext, 1000.0 * math.log10(0.3), places=4)

        # Mirror-clone 1:1 parity detection
        rng = np.random.default_rng(103)
        X_base_tr = rng.normal(size=(160, 6))
        X_base_te = rng.normal(size=(40, 6))
        X_mir = rng.normal(size=(30, 6))
        alerts, diag = audit_split_leakage(
            np.vstack([X_base_tr, X_mir]),
            np.vstack([X_base_te, X_mir]),
        )
        rules = {a.rule_id for a in alerts}
        self.assertIn("LEAK-001", rules)
        self.assertIn("LEAK-002", rules)
        self.assertEqual(diag["multiplicity_parity"]["m2_mirror"], 30)
        self.assertLess(diag["multiplicity_parity"]["m2_log10_p"], -10.0)

    def test_conflicting_labels(self) -> None:
        rng = np.random.default_rng(104)
        X_tr = rng.normal(size=(120, 6))
        y_tr = rng.binomial(1, 0.5, size=120).astype(np.float64)
        X_te = rng.normal(size=(50, 6))
        y_te = rng.binomial(1, 0.5, size=50).astype(np.float64)
        # Same continuous profile carries opposite labels across splits
        X_te[0] = X_tr[0]
        y_te[0] = 1.0 - y_tr[0]
        alerts, diag = audit_split_leakage(X_tr, X_te, y_tr, y_te)
        self.assertTrue(any(a.rule_id == "LEAK-003" for a in alerts))
        self.assertEqual(diag["cross_split_conflicts"], 1)

    def test_confusion_matrix_unseen_bounds(self) -> None:
        # Published: N_neg=100 (TN=60, FP=40), N_pos=100 (TP=85, FN=15)
        # Known leaked: 50 negatives, 40 positives => 50 unseen neg, 60 unseen pos
        res = confusion_matrix_unseen_bounds(
            tn=60, fp=40, fn=15, tp=85, n_leak_neg=50, n_leak_pos=40
        )
        # Unseen specificity: model-free upper = min(60, 50)/50 = 1.0, lower/mem = (60-50)/50 = 0.20
        self.assertAlmostEqual(res["specificity"]["published"], 0.60)
        self.assertAlmostEqual(res["specificity"]["model_free_upper_bound"], 1.0)
        self.assertAlmostEqual(res["specificity"]["model_free_lower_bound"], 0.20)
        self.assertAlmostEqual(res["specificity"]["memorization_upper_bound"], 0.20)
        self.assertAlmostEqual(res["specificity"]["non_inferiority_upper_bound"], 0.60)

        # Case where model-free upper bound is strictly < 1.0: TN=30, FP=70, n_leak_neg=40 (60 unseen)
        res2 = confusion_matrix_unseen_bounds(
            tn=30, fp=70, fn=10, tp=90, n_leak_neg=40, n_leak_pos=30
        )
        self.assertAlmostEqual(res2["specificity"]["model_free_upper_bound"], 30.0 / 60.0)
        self.assertAlmostEqual(res2["specificity"]["model_free_lower_bound"], 0.0)

        # 100% leaked negatives -> unidentifiable
        res3 = confusion_matrix_unseen_bounds(tn=80, fp=20, fn=10, tp=90, n_leak_neg=100, n_leak_pos=0)
        self.assertFalse(res3["specificity"]["identifiable"])
        self.assertIsNone(res3["specificity"]["model_free_upper_bound"])

        # Validation error if n_leak > class count
        with self.assertRaises(ValueError):
            confusion_matrix_unseen_bounds(tn=10, fp=5, fn=5, tp=10, n_leak_neg=20, n_leak_pos=0)

    def test_xlsx_and_csv_split_files(self) -> None:
        rng = np.random.default_rng(105)
        X_tr = rng.normal(size=(80, 5))
        X_te = rng.normal(size=(40, 5))
        X_te[:15] = X_tr[:15]
        headers = ["f1", "f2", "f3", "f4", "f5", "label"]
        tr_rows = [headers] + [[*row.tolist(), int(i % 2)] for i, row in enumerate(X_tr)]
        te_rows = [headers] + [[*row.tolist(), int(i % 2)] for i, row in enumerate(X_te)]

        with tempfile.TemporaryDirectory() as tmp:
            wb_path = Path(tmp) / "splits.xlsx"
            write_xlsx(wb_path, {"Train_Cohort": tr_rows, "Test_Cohort": te_rows})
            alerts = audit_xlsx_splits(wb_path)
            self.assertTrue(any(a.rule_id == "LEAK-001" for a in alerts))

            tr_csv = Path(tmp) / "data_train.csv"
            te_csv = Path(tmp) / "data_test.csv"
            tr_csv.write_text("\n".join(",".join(map(str, r)) for r in tr_rows), encoding="utf-8")
            te_csv.write_text("\n".join(",".join(map(str, r)) for r in te_rows), encoding="utf-8")
            c_alerts, _ = audit_tabular_split_files(tr_csv, te_csv)
            self.assertTrue(any(a.rule_id == "LEAK-001" for a in c_alerts))


class TestModule2EvalCheck(unittest.TestCase):
    def test_all_six_code_patterns_and_clean_controls(self) -> None:
        # (a) Binarized clinical score
        doc_a = CodeDocument(
            "a.py", "python",
            "score_bin = (cha2ds2_vasc >= 2).astype(int)\nauc = roc_auc_score(y_test, score_bin)\n",
        )
        self.assertEqual({a.rule_id for a in scan_code_document(doc_a)}, {"EVAL-001"})

        # Clean clinical score binarized only for confusion_matrix
        doc_a_clean = CodeDocument(
            "a_ok.py", "python",
            "auc = roc_auc_score(y_test, cha2ds2_vasc)\n"
            "score_bin = (cha2ds2_vasc >= 2).astype(int)\n"
            "cm = confusion_matrix(y_test, score_bin)\n",
        )
        self.assertEqual(scan_code_document(doc_a_clean), [])

        # (b) Hardcoded DCA
        doc_b = CodeDocument(
            "b.py", "python",
            "y_pred = (y_prob >= 0.5).astype(int)\n"
            "net_benefits = []\n"
            "for pt in thresholds:\n"
            "    tp = ((y_pred == 1) & (y == 1)).sum()\n"
            "    fp = ((y_pred == 1) & (y == 0)).sum()\n"
            "    net_benefits.append(tp/n - fp/n * (pt/(1-pt)))\n",
        )
        self.assertEqual({a.rule_id for a in scan_code_document(doc_b)}, {"EVAL-002"})

        # (c) Binary prediction to Brier score
        doc_c = CodeDocument(
            "c.py", "python",
            "clf = RandomForestClassifier().fit(X_tr, y_tr)\n"
            "b = brier_score_loss(y_te, clf.predict(X_te))\n",
        )
        self.assertEqual({a.rule_id for a in scan_code_document(doc_c)}, {"EVAL-003"})

        # (d) Pre-split SMOTE leakage vs Pipeline SMOTE clean
        doc_d = CodeDocument(
            "d.py", "python",
            "X_res, y_res = SMOTE().fit_resample(X, y)\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res)\n",
        )
        self.assertEqual({a.rule_id for a in scan_code_document(doc_d)}, {"EVAL-004"})

        doc_d_clean = CodeDocument(
            "d_ok.py", "python",
            "pipe = Pipeline([('smote', SMOTE()), ('clf', LogisticRegression())])\n"
            "scores = cross_val_score(pipe, X, y, cv=5)\n",
        )
        self.assertEqual(scan_code_document(doc_d_clean), [])

        # (e) Immortal-time bias (Python & R) vs time-varying Cox clean
        doc_e = CodeDocument(
            "e.R", "r",
            "library(survival)\nfit <- survfit(Surv(time, event) ~ received_adjuvant_chemo, data = df)\n",
        )
        self.assertEqual({a.rule_id for a in scan_code_document(doc_e)}, {"EVAL-005"})

        doc_e_clean = CodeDocument(
            "e_ok.R", "r",
            "library(survival)\nfit <- coxph(Surv(tstart, tstop, event) ~ received_adjuvant_chemo + age, data = df)\n",
        )
        self.assertEqual(scan_code_document(doc_e_clean), [])

        # (f) Test-set checkpoint selection
        doc_f = CodeDocument(
            "f.py", "python",
            "model.fit(X_train, y_train, eval_set=[(X_test, y_test)])\n",
        )
        self.assertEqual({a.rule_id for a in scan_code_document(doc_f)}, {"EVAL-006"})

    def test_tabular_dca_checker(self) -> None:
        pts = np.linspace(0.05, 0.85, 17)
        odds = pts / (1.0 - pts)
        prev = 0.30
        ta = prev - (1.0 - prev) * odds

        # Clean continuous model DCA
        rng = np.random.default_rng(7)
        y = rng.binomial(1, prev, size=1000)
        emp_p = float(np.mean(y))
        probs = 1.0 / (1.0 + np.exp(-np.where(y == 1, rng.normal(1.1, 1.0, 1000), rng.normal(-1.0, 1.0, 1000))))
        nbs_clean = [np.mean((probs >= t) & (y == 1)) - np.mean((probs >= t) & (y == 0)) * (t / (1 - t)) for t in pts]
        alerts_ok, _ = check_dca_curve(pts, nbs_clean, prevalence=emp_p, treat_all=emp_p - (1 - emp_p) * odds)
        self.assertEqual(alerts_ok, [])

        # Flat constant Net Benefit (using percentage thresholds [5, 85] to test auto-normalization)
        alerts_flat, _ = check_dca_curve(pts * 100.0, np.full_like(pts, 0.18), prevalence=prev)
        self.assertTrue(any(a.rule_id == "EVAL-002" for a in alerts_flat))

        # Exact linear-in-odds single-threshold DCA
        nb_fixed = 0.24 - 0.12 * odds
        alerts_fixed, _ = check_dca_curve(pts, nb_fixed, prevalence=prev, treat_all=ta)
        self.assertTrue(any(a.rule_id == "EVAL-002" for a in alerts_fixed))

        # Table wrapper
        tbl = TableData(
            name="DCA",
            headers=["threshold", "treat_all", "model_net_benefit"],
            rows=[[float(t), float(a), 0.18] for t, a in zip(pts, ta)],
        )
        self.assertTrue(any(a.rule_id == "EVAL-002" for a in audit_dca_table(tbl)))


class TestModule3StatCheck(unittest.TestCase):
    def test_exact_statistical_helpers(self) -> None:
        # Exact binomial: k=5, n=5, p0=0.5 => two-sided = 2 * (1/32) = 0.0625
        res_b = exact_binomial_test(5, 5, p0=0.5, alternative="two-sided")
        self.assertAlmostEqual(res_b["p_value"], 0.0625, places=10)

        # Exact McNemar: b=10, c=0 => 2 * 2^-10 = 2^-9 = 1/512
        res_m = exact_mcnemar_test(10, 0)
        self.assertAlmostEqual(res_m["p_value"], 1.0 / 512.0, places=10)
        p_min, _ = min_mcnemar_p_for_net_diff(10)
        self.assertAlmostEqual(p_min, 1.0 / 512.0, places=10)

        # Paired-Z sensitivity contradiction
        pz = paired_z_sensitivity_check(delta=0.008, se1=0.018, se2=0.018, p_reported=1e-12)
        self.assertTrue(pz["contradicts_reported_p"])
        pz_ok = paired_z_sensitivity_check(delta=0.08, se1=0.015, se2=0.015, p_reported=1e-6)
        self.assertFalse(pz_ok["contradicts_reported_p"])

    def test_statcheck_rules_and_xlsx_file(self) -> None:
        # Fixed-width placeholder CIs
        tbl_fixed = TableData(
            name="Fixed",
            headers=["Cohort", "N", "AUROC"],
            rows=[
                ["C1", 200, "0.720 ± 0.015"],
                ["C2", 450, "0.785 ± 0.015"],
                ["C3", 900, "0.840 ± 0.015"],
                ["C4", 1200, "0.892 ± 0.015"],
                ["C5", 300, "0.755 ± 0.015"],
                ["C6", 1500, "0.910 ± 0.015"],
            ],
        )
        alerts_f, _ = audit_table_stats(tbl_fixed)
        self.assertTrue(any(a.rule_id == "STAT-001" for a in alerts_f))

        # Zero-width CI on interior estimate
        tbl_zero = TableData(
            name="Zero",
            headers=["Model", "AUC", "CI_low", "CI_high"],
            rows=[["M1", 0.85, 0.85, 0.85], ["M2", 0.82, 0.79, 0.85]],
        )
        alerts_z, _ = audit_table_stats(tbl_zero)
        self.assertTrue(any(a.rule_id == "STAT-002" for a in alerts_z))

        # Cross-sheet sqrt(1000) SE-of-bootstrap contradiction in real .xlsx
        t1_rows = [
            ["Task", "AUROC"],
            ["Mortality", "0.8600 (0.8200-0.9000)"],
            ["Readmission", "0.7900 (0.7400-0.8400)"],
        ]
        # Width divided by sqrt(1000) ~ 31.62
        t2_rows = [
            ["Task", "AUROC"],
            ["Mortality", "0.8600 (0.8587-0.8613)"],
            ["Readmission", "0.7900 (0.7884-0.7916)"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            wb_p = Path(tmp) / "source_data.xlsx"
            write_xlsx(wb_p, {"Fig2a": t1_rows, "SuppTable2": t2_rows})
            x_alerts, x_diag = audit_stat_file(wb_p)
            self.assertTrue(any(a.rule_id == "STAT-003" for a in x_alerts))
            self.assertEqual(x_diag["cross_table"]["contradictions"][0]["matched_sqrt_B"], 1000)

        # Impossible 5-fold Wilcoxon P < 0.03125
        tbl_w = TableData(
            name="Wilcox",
            headers=["Task", "p_value", "method"],
            rows=[["T1", 0.002, "5-fold paired Wilcoxon"]],
        )
        alerts_w, _ = audit_table_stats(tbl_w)
        self.assertTrue(any(a.rule_id == "STAT-004" for a in alerts_w))


class TestModule4DeployCheck(unittest.TestCase):
    def test_axioms_encoding_monotonicity_and_coefficients(self) -> None:
        rng = np.random.default_rng(202)
        fnames = ["age", "sex", "egfr", "albumin", "creatinine"]
        X = np.column_stack([
            rng.uniform(45, 85, 80),
            rng.binomial(1, 0.5, 80),
            rng.uniform(20, 105, 80),
            rng.uniform(2.8, 4.7, 80),
            rng.uniform(0.7, 3.0, 80),
        ])

        def _sig(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

        # Clean hierarchical calculator
        def fn_all(arr: np.ndarray) -> np.ndarray:
            lp = -1.5 + 0.05 * (arr[:, 0] - 60) + 0.3 * arr[:, 1] - 0.03 * (arr[:, 2] - 65) - 0.9 * (arr[:, 3] - 4.0) + 0.4 * (arr[:, 4] - 1.2)
            return _sig(lp)

        def fn_cv(arr: np.ndarray) -> np.ndarray:
            return fn_all(arr) * 0.55

        alerts_clean, _ = audit_risk_calculator(
            {"all_cause": fn_all, "cv": fn_cv},
            fnames,
            X,
            hierarchy_pairs=[("cv", "all_cause")],
        )
        self.assertEqual(alerts_clean, [])

        # Violate hierarchy + monotonicity (abs(coef) bug!) + bounds
        def fn_buggy_cause(arr: np.ndarray) -> np.ndarray:
            # All signs forced positive -> egfr and albumin inverted! Also exceeds fn_all!
            lp = -0.2 + 0.05 * (arr[:, 0] - 60) + 0.3 * (arr[:, 1] == 2) + 0.03 * (arr[:, 2] - 65) + 0.9 * (arr[:, 3] - 4.0) + 0.4 * (arr[:, 4] - 1.2)
            return _sig(lp)

        alerts_bug, _ = audit_risk_calculator(
            {"all_cause": fn_all, "cv": fn_buggy_cause},
            fnames,
            X,
            hierarchy_pairs=[("cv", "all_cause")],
        )
        rules = {a.rule_id for a in alerts_bug}
        self.assertIn("DEPLOY-001", rules)
        self.assertIn("DEPLOY-002", rules)
        self.assertIn("DEPLOY-003", rules)

        # Direct linear coefficient sign audit
        c_alerts, _ = audit_linear_coefficients({"age": 0.04, "egfr": 0.025, "albumin": 0.45})
        self.assertEqual({a.rule_id for a in c_alerts}, {"DEPLOY-003"})

    def test_static_abs_coef_detector(self) -> None:
        # Buggy Python & R abs(coef)
        py_bug = CodeDocument("calc.py", "python", "weights = np.abs(model.coef_[0])\nrisk = sigmoid(X @ weights)\n")
        self.assertEqual({a.rule_id for a in scan_deploy_code_document(py_bug)}, {"DEPLOY-004"})

        r_bug = CodeDocument("calc.R", "r", "w <- abs(coef(cox_fit))\nlp <- X %*% w\n")
        self.assertEqual({a.rule_id for a in scan_deploy_code_document(r_bug)}, {"DEPLOY-004"})

        # Clean feature importance sorting must NOT trigger
        py_ok = CodeDocument("imp.py", "python", "top_idx = np.argsort(np.abs(model.coef_[0]))[::-1]\n")
        self.assertEqual(scan_deploy_code_document(py_ok), [])


class TestScannerAndCLI(unittest.TestCase):
    def test_unified_scanner_and_cli_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Leaked train/test CSV pair
            rng = np.random.default_rng(303)
            X_tr = rng.normal(size=(60, 5))
            X_te = rng.normal(size=(30, 5))
            X_te[:12] = X_tr[:12]
            hdr = "f1,f2,f3,f4,f5,y"
            (root / "cohort_train.csv").write_text(
                hdr + "\n" + "\n".join(",".join(f"{v:.6f}" for v in row) + f",{i%2}" for i, row in enumerate(X_tr)),
                encoding="utf-8",
            )
            (root / "cohort_test.csv").write_text(
                hdr + "\n" + "\n".join(",".join(f"{v:.6f}" for v in row) + f",{i%2}" for i, row in enumerate(X_te)),
                encoding="utf-8",
            )
            # 2. Pipeline script with Module 2 & Module 4 defects
            (root / "pipeline.py").write_text(
                "X_res, y_res = SMOTE().fit_resample(X, y)\n"
                "X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res)\n"
                "w = np.abs(cox_coef)\n"
                "risk = X_te @ w\n",
                encoding="utf-8",
            )
            # 3. Source Data workbook with Module 3 zero-width CI
            write_xlsx(
                root / "source_data.xlsx",
                {"Table1": [["Model", "AUC", "CI_low", "CI_high"], ["M1", 0.88, 0.88, 0.88]]},
            )

            report = audit_directory(root)
            self.assertEqual(report.version, __version__)
            by_mod = report.summary["by_module"]
            self.assertGreater(by_mod["leakage"], 0)
            self.assertGreater(by_mod["evalcheck"], 0)
            self.assertGreater(by_mod["statcheck"], 0)
            self.assertGreater(by_mod["deploycheck"], 0)

            # Test CLI via subprocess `python -m clinaudit scan <dir> --out report.json --fail-on-critical`
            out_json = root / "report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clinaudit",
                    "scan",
                    str(root),
                    "--out",
                    str(out_json),
                    "--fail-on-critical",
                    "--quiet",
                ],
                cwd=str(_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertTrue(out_json.exists())
            parsed = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed["tool"], "clinaudit")
            self.assertTrue(parsed["summary"]["flagged"])
            self.assertGreaterEqual(parsed["summary"]["by_severity"]["CRITICAL"], 4)

            # Clean directory returns exit code 0
            clean_dir = root / "clean_proj"
            clean_dir.mkdir()
            (clean_dir / "clean.py").write_text(
                "pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])\n"
                "auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])\n",
                encoding="utf-8",
            )
            proc_ok = subprocess.run(
                [sys.executable, "-m", "clinaudit", "scan", str(clean_dir), "--fail-on-critical", "--quiet"],
                cwd=str(_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_ok.returncode, 0)

            # Verify audit_files directly
            rep_files = audit_files([root / "pipeline.py"])
            self.assertEqual(len(rep_files.files_scanned), 1)
            self.assertGreaterEqual(rep_files.summary["total_alerts"], 2)

    def test_skipped_files_and_parse_error_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
            (root / "archive.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
            (root / "corrupt.xlsx").write_bytes(b"not a valid zip xlsx")
            rep = audit_directory(root)
            self.assertEqual(len(rep.files_scanned), 2)
            self.assertEqual(len(rep.files_skipped), 1)
            self.assertTrue(any(p.endswith("archive.zip") for p in rep.files_skipped))
            self.assertGreaterEqual(len(rep.parse_errors), 1)
            self.assertTrue(any(e["file"].endswith("corrupt.xlsx") for e in rep.parse_errors))
            d = rep.to_dict()
            self.assertEqual(d["summary"]["files_skipped"], 1)
            self.assertGreaterEqual(d["summary"]["parse_errors"], 1)


if __name__ == "__main__":
    unittest.main()
