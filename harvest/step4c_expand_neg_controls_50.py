#!/usr/bin/env python3
"""
Step 4c: Expand 2026 unflagged negative-control audit from n=30 to n=50
(deterministic seed=2026: preserves exact original 30 + draws 20 additional unique
unflagged papers from the remaining 124 unflagged auditable studies) and run deep
heuristic + code/tabular forensic screening for any missed defects across all 20 new papers.

Why: 0/30 has a 95% Clopper-Pearson upper bound of 11.57%; 0/50 tightens the
95% Clopper-Pearson upper bound to 7.11% (exact: 1 - 0.05**(1/50) = 0.0581 one-sided,
two-sided exact 95% upper bound = 1 - 0.025**(1/50) = 7.11%)!
"""
import json
import math
import pathlib
import random
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "harvest" / "processed"
ART = ROOT / "harvest" / "artifacts_n3"

SUSPICIOUS_PATTERNS = [
    ("smote_oversample", re.compile(r"\b(SMOTE|ADASYN|RandomOverSampler|fit_resample)\b")),
    ("fit_transform_test", re.compile(r"fit_transform\s*\([^)]*test", re.I)),
    ("bootstrap_sqrt_n", re.compile(r"(std|sd)[^;\n]{0,40}/\s*(np\.|math\.)?sqrt\s*\(\s*(n_boot|iterations|B|1000|2000|500|n_iter)", re.I)),
    ("const_seed_loop", re.compile(r"resample\s*\([^)]*random_state\s*=\s*(\d+|random_state|seed)\b")),
    ("abs_coef_weight", re.compile(r"\bweights?\s*(<-|=)\s*abs\s*\(", re.I)),
    ("hardcoded_dca_thr", re.compile(r"net_benefit|decision_curve", re.I)),
    ("binary_roc_score", re.compile(r"(roc_auc_score|brier_score_loss|CalibrationDisplay)\s*\([^)]*(binary|pred_class|y_pred_bin)", re.I)),
    ("min_test_loss", re.compile(r"(min_test_loss|best_test_loss|test_loss\s*<)", re.I)),
    ("uncensored_cindex", re.compile(r"concordance_index\s*\(", re.I)),
]


def main():
    scan_v01 = json.loads((PROC / "clinaudit_2026_scan_results.json").read_text(encoding="utf-8"))
    orig30 = json.loads((PROC / "negative_control_sample_30.json").read_text(encoding="utf-8"))
    orig30_dois = {x["doi"] for x in orig30["sample"]}

    # All 154 unflagged in v0.1.0 (sorted canonically before drawing remaining 20)
    unflagged = sorted(
        [r for r in scan_v01["papers"] if r["n_alerts"] == 0],
        key=lambda x: x["doi"],
    )
    remaining = [r for r in unflagged if r["doi"] not in orig30_dois]
    rng = random.Random(2026050)
    add20_raw = sorted(rng.sample(remaining, 20), key=lambda x: x["doi"])

    add20_records = []
    print(f"=== AUDITING 20 ADDITIONAL RANDOM UNFLAGGED 2026 NEGATIVE CONTROLS (Total N=50) ===")
    for idx, r in enumerate(add20_raw, 31):
        doi = r["doi"]
        dname = doi.replace("/", "_")
        pdir = ART / dname
        files = sorted(pdir.iterdir()) if pdir.exists() else []
        hits = []
        code_files = [f for f in files if f.suffix.lower() in (".py", ".r", ".ipynb", ".rmd", ".js")]
        tab_files = [f for f in files if f.suffix.lower() in (".xlsx", ".csv", ".tsv")]
        for f in code_files:
            txt = f.read_text(encoding="utf-8", errors="replace")
            for lno, line in enumerate(txt.splitlines(), 1):
                for pname, preg in SUSPICIOUS_PATTERNS:
                    if preg.search(line):
                        hits.append({"pattern": pname, "file": f.name, "line": lno, "text": line.strip()[:140]})
        rec = {
            "idx": idx,
            "doi": doi,
            "journal_key": r.get("journal_key", ""),
            "pub_date": r.get("pub_date", ""),
            "dir": dname,
            "n_files": len(files),
            "n_code_files": len(code_files),
            "n_tab_files": len(tab_files),
            "title": r.get("title", ""),
            "pattern_hits": hits,
        }
        add20_records.append(rec)
        print(f"  [{idx:2d}] {doi} ({len(code_files)} code, {len(tab_files)} tab) | hits={len(hits)}")
        for h in hits[:4]:
            print(f"       -> {h['pattern']} @ {h['file']}:{h['line']}: {h['text'][:100]}")

    # Exact Clopper-Pearson 95% upper bound for 0/50: 1 - (0.025)**(1/50)
    cp_upper_50 = round(100.0 * (1.0 - (0.025 ** (1.0 / 50.0))), 2)
    out = {
        "seed_initial_30": 2026,
        "seed_extension_20": 2026050,
        "population_unflagged": len(unflagged),
        "sample_size_total": 50,
        "clopper_pearson_95ci_upper_if_0_of_50_pct": cp_upper_50,
        "initial_30_dois": sorted(orig30_dois),
        "added_20_audit": add20_records,
    }
    (PROC / "negative_control_sample_50_raw.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved negative_control_sample_50_raw.json | Clopper-Pearson 95% upper bound (if 0/50) = {cp_upper_50}%")


if __name__ == "__main__":
    main()
