"""Targeted unit tests for ClinAudit v0.3.0 statistical rules, parser upgrades, and noise filters."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clinaudit import (  # noqa: E402
    TableData,
    audit_cross_table_ci_consistency,
    audit_table_stats,
    read_tables,
    write_xlsx,
)


class TestV030StatCheckUpgrades(unittest.TestCase):
    def test_stacked_subtable_splitting_prevents_header_spillover(self) -> None:
        sheets = {
            "MultiPanel": [
                ["Figure 1A: Model Performance"],
                ["Metric", "Estimate", "95% CI MIN", "95% CI MAX"],
                ["AUROC", 0.91, 0.87, 0.95],
                ["Specificity", 0.9625, 0.9641, 0.9682],
                [None, None, None, None],
                ["Figure 1B: Longitudinal measurements"],
                ["SubjectID", "Day1", "Day7", "Day14"],
                ["S001", 120.5, 98.2, 45.1],
                ["S002", 88.0, 74.0, 32.0],
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "stacked.xlsx"
            write_xlsx(p, sheets)
            tbls = read_tables(p)
            self.assertEqual(len(tbls), 2)
            self.assertEqual(tbls[0].sheet_base_name, "MultiPanel")
            self.assertEqual(tbls[1].sheet_base_name, "MultiPanel")
            alerts_0, _ = audit_table_stats(tbls[0])
            alerts_1, _ = audit_table_stats(tbls[1])
            self.assertTrue(any(a.rule_id == "STAT-001" and a.priority == "P0" for a in alerts_0))
            self.assertEqual(alerts_1, [])

    def test_est_outside_ci_inline_and_hi_lo_order(self) -> None:
        t_inline = TableData(
            name="InlineOut",
            headers=["Metric", "Model_A"],
            rows=[["F1 score", "0.79 [0.48, 0.50]"], ["Accuracy", "0.85 [0.81, 0.89]"]],
        )
        alerts_in, _ = audit_table_stats(t_inline)
        self.assertTrue(
            any(a.rule_id == "STAT-001" and a.priority == "P0" and "outside" in a.title.lower() for a in alerts_in)
        )

        t_rev = TableData(
            name="RevOrder",
            headers=["FeatureSet", "coefficient", "95% CI coefficient upper bound", "95% CI coefficient lower bound"],
            rows=[
                ["set_1", 0.51, 1.00, 0.17],
                ["set_2", 0.67, 0.0, 0.0],
            ],
        )
        alerts_rev, _ = audit_table_stats(t_rev)
        self.assertTrue(any(a.rule_id == "STAT-001" and a.priority == "P0" for a in alerts_rev))

    def test_out_of_domain_ci_bounds(self) -> None:
        t_dom = TableData(
            name="DomCI",
            headers=["Model", "AUROC", "CI_low", "CI_high"],
            rows=[["M1", 0.99, 0.95, 1.03], ["M2", 0.85, 0.80, 0.90]],
        )
        alerts, _ = audit_table_stats(t_dom)
        self.assertTrue(
            any(a.rule_id == "STAT-001" and a.priority == "P0" and "domain" in a.title.lower() for a in alerts)
        )

    def test_stat006_always_on_even_for_omics(self) -> None:
        t_neg = TableData(
            name="Proteomics_MWU",
            headers=["AssayId", "Protein", "p", "FDR", "n"],
            rows=[
                ["A-001", "Marker_A", 0.0021, -0.012, 80],
                ["A-002", "Marker_B", 0.00015, 0.0045, 80],
                ["A-003", "Marker_C", 0.025, 0.14, 80],
            ],
        )
        alerts_neg, _ = audit_table_stats(t_neg)
        self.assertTrue(any(a.rule_id == "STAT-006" and a.priority == "P0" for a in alerts_neg))

        t_tiny = TableData(
            name="Cluster_Corr",
            headers=["cluster1", "cluster2", "n_donors", "p.value", "fdr"],
            rows=[["C-1", "C-2", 4, 0.0, 0.0], ["C-3", "C-4", 28, 0.001, 0.01]],
        )
        alerts_tiny, _ = audit_table_stats(t_tiny)
        self.assertTrue(any(a.rule_id == "STAT-006" and a.priority == "P0" for a in alerts_tiny))

        t_err = TableData(
            name="BenchPerf",
            headers=["Method", "f1_score", "p_value"],
            rows=[["RefModel", 0.89, "#NUM!"], ["ModelB", 0.72, 0.001]],
        )
        alerts_err, _ = audit_table_stats(t_err)
        self.assertTrue(any(a.rule_id == "STAT-006" and a.priority == "P0" for a in alerts_err))

    def test_nested_model_or_vs_diff_contradiction_p0(self) -> None:
        t_nest = TableData(
            name="NestedModels",
            headers=[
                "Outcome", "Marker", "Model", "St. OR", "St. Lower CI", "St. Upper CI",
                "p-value", "AUC", "Difference of Model 2 to minimal model",
                "p-value for difference to minimal model", "n_min_max",
            ],
            rows=[
                ["Outcome_1", "Biomarker_1", 2, 2.11, 1.62, 2.76, 3.0e-7, 0.807, None, None, "450-460"],
                [None, None, "Minimal Model", 1.10, 0.87, 1.40, 0.405, 0.569, 0.24, 1.0e-42, "450-460"],
            ],
        )
        alerts, _ = audit_table_stats(t_nest)
        self.assertTrue(
            any(a.rule_id == "STAT-004" and a.priority == "P0" and "nested-model" in a.title.lower() for a in alerts)
        )

    def test_algebraic_self_consistency_and_fixed_decimal_zero_p_filters(self) -> None:
        t_alg = TableData(
            name="RegModel",
            headers=["Model", "AUC", "Beta", "SE", "p_value"],
            rows=[["M1", 0.82, 1.20, 0.10, 3.5e-33], ["M2", 0.78, 0.30, 0.10, 0.0027]],
        )
        alerts_alg, _ = audit_table_stats(t_alg)
        self.assertFalse(any(a.rule_id == "STAT-004" for a in alerts_alg))

        rows_rnd = [[f"M{i}", 0.80, f"{0.0001 * (i + 1):.4f}"] for i in range(60)]
        rows_rnd.append(["M_top", 0.92, "0.0000"])
        t_rnd = TableData(
            name="RoundedTable",
            headers=["Model", "AUC", "p_value"],
            rows=[[r[0], r[1], float(r[2])] for r in rows_rnd],
            raw_rows=[[r[0], str(r[1]), r[2]] for r in rows_rnd],
        )
        alerts_rnd, _ = audit_table_stats(t_rnd)
        self.assertFalse(any(a.rule_id == "STAT-004" for a in alerts_rnd))

    def test_legitimate_zero_width_and_euro_comma_and_cross_table_guards(self) -> None:
        t_clean = TableData(
            name="CoxClean",
            headers=["Category", "n (%)", "HR", "CI_low", "CI_high", "p_value"],
            rows=[
                ["Q1 (Reference)", "7 (4,2)", 1.0, 1.0, 1.0, None],
                ["Q2", "12 (7,5)", 1.25, 1.05, 1.48, 0.012],
                ["Continuous per 100 units", "50 (20,1)", 1.0, 1.0, 1.0, 0.65],
            ],
            raw_rows=[
                ["Q1 (Reference)", "7 (4,2)", "1.00", "1.00", "1.00", ""],
                ["Q2", "12 (7,5)", "1.25", "1.05", "1.48", "0.012"],
                ["Continuous per 100 units", "50 (20,1)", "1.00", "1.00", "1.00", "0.65"],
            ],
        )
        alerts_c, _ = audit_table_stats(t_clean)
        self.assertEqual(alerts_c, [])

        t_3y = TableData(name="Surv_3y", headers=["Model", "AUROC"], rows=[["RiskModel", "0.840 (0.800-0.880)"]])
        t_8y = TableData(name="Surv_8y", headers=["Model", "AUROC"], rows=[["RiskModel", "0.840 (0.835-0.845)"]])
        alerts_x, _ = audit_cross_table_ci_consistency([t_3y, t_8y])
        self.assertEqual(alerts_x, [])

    def test_v031_anatomy_compound_prefix_non_estimate_and_domain_guards(self) -> None:
        # 1. Anatomy/panel/cutoff lower & upper columns must NOT trigger STAT-001
        t_anat = TableData(
            name="AnatClean",
            headers=["ID", "Lower Respiratory Diseases", "Upper Respiratory Diseases", "Upper body", "Lower body", "count_lower_cutoff", "count_upper_cutoff"],
            rows=[["S1", 0.72, 0.65, 8.2, 12.4, 500, 0]],
        )
        alerts_a, _ = audit_table_stats(t_anat)
        self.assertEqual(alerts_a, [])

        # 2. Compound group prefix (emt-high_CI_lower / emt-high_CI_upper) must pair cleanly without dual-match
        t_comp = TableData(
            name="KMComp",
            headers=["time", "emt-high_survival_probability", "emt-high_CI_lower", "emt-high_CI_upper", "emt-low_survival_probability", "emt-low_CI_lower", "emt-low_CI_upper"],
            rows=[[0.5, 0.99, 0.95, 0.999, 0.98, 0.94, 0.998]],
        )
        alerts_comp, _ = audit_table_stats(t_comp)
        self.assertEqual(alerts_comp, [])

        # 3. Adjacent _se / _std / abs_ / p columns must NOT bind as point estimate, and (20,484) thousands counts must NOT parse as CI
        t_est = TableData(
            name="MREstClean",
            headers=["Protein", "IVW_b", "IVW_se", "IVW_lowCI", "IVW_highCI", "ROC-AUC_mean", "ROC-AUC_std", "Proportion (N)"],
            rows=[["P1", 0.06, 0.009, 0.042, 0.077, 0.902, 0.031, "0.9247 (20,484)"]],
        )
        alerts_est, _ = audit_table_stats(t_est)
        self.assertEqual(alerts_est, [])

        # 4. GSEA gene-set size N + Survival HR + mean ± SD + mixed [0,1]/[0,100]% scales must NOT FP
        t_gsea = TableData(
            name="GSEAClean",
            headers=["Gene Set", "N", "ES", "NES", "p-value", "FDR"],
            rows=[["PATHWAY_A", 9, -0.84, -2.11, 0.0, 0.0]],
        )
        self.assertEqual(audit_table_stats(t_gsea)[0], [])

        t_dom = TableData(
            name="DomainClean",
            headers=["Gene", "Impact on survival_HR (95% CI)", "AUC", "PPV", "PPV 95%CI"],
            rows=[["G1", "8.402 (0.227-311.556)", "0.998 ± 0.004", 0.971, "(74.0 - 99.9)"]],
        )
        self.assertEqual(audit_table_stats(t_dom)[0], [])

        # 5. Positive recall lock: real HR outside CI, unclipped NPV/Sens CI > 1, and p_val > 1 MUST stay P0!
        t_real = TableData(
            name="RealDefectsLock",
            headers=["Marker", "HR (95% CI)", "sens", "sens CI lower", "sens CI upper", "p_val"],
            rows=[
                ["Stage", "-1.4 (0.12-0.88)", 0.9200, 0.8100, 1.0300, 1.0450],
                ["Subgroup", "1.2 (0.9-1.5)", 0.85, 0.80, 0.90, 0.04],
                ["Subgroup2", "1.1 (0.8-1.4)", 0.82, 0.77, 0.87, 0.12],
            ],
        )
        alerts_real, _ = audit_table_stats(t_real)
        p0_rules = {a.rule_id for a in alerts_real if a.priority == "P0"}
        self.assertIn("STAT-001", p0_rules)
        self.assertIn("STAT-006", p0_rules)


if __name__ == "__main__":
    unittest.main()
