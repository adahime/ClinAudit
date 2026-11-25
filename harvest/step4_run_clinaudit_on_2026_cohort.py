#!/usr/bin/env python3
"""
Step 4: Run `clinaudit` v0.1.0 across ALL 174 downloaded 2026 auditable clinical ML papers
(2,678 code & tabular Source Data files) in `harvest/artifacts_n3/`.

Outputs:
  - `harvest/processed/clinaudit_2026_scan_results.json` (full structured per-paper scan reports)
  - `harvest/processed/clinaudit_2026_alerts.csv` (flat alert table for adversarial review)
  - `harvest/processed/negative_control_sample_30.json` (deterministic seed=2026 random sample of 30 unflagged papers)
"""
import csv
import json
import pathlib
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinaudit.scanner import audit_directory

PROC_DIR = ROOT / "harvest" / "processed"
ART_DIR = ROOT / "harvest" / "artifacts_n3"


def main():
    manifest = json.loads((PROC_DIR / "n3_downloaded_manifest.json").read_text(encoding="utf-8"))
    eligible_map = {
        r["doi"]: r
        for r in json.loads((PROC_DIR / "eligible_2026_verified.json").read_text(encoding="utf-8"))
    }
    targets = [r for r in manifest["records"] if r["n_files"] > 0]
    print(f"Scanning {len(targets)} downloaded 2026 papers ({manifest['total_extracted_files']} files) with ClinAudit v0.1.0...", flush=True)

    per_paper = []
    all_alerts = []
    t0 = time.time()

    for idx, rec in enumerate(targets, 1):
        doi = rec["doi"]
        jkey = rec["journal_key"]
        pdir = ART_DIR / rec["dir"]
        tp0 = time.perf_counter()
        err = None
        alerts = []
        scanned_files = 0
        try:
            rep = audit_directory(str(pdir))
            rep_d = rep.to_dict() if hasattr(rep, "to_dict") else rep
            alerts = rep_d.get("alerts") or []
            scanned_files = rep_d.get("files_scanned", rec["n_files"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        elapsed_ms = round((time.perf_counter() - tp0) * 1000.0, 2)

        sev_counts = Counter(a.get("severity", "UNKNOWN") for a in alerts)
        mod_counts = Counter(a.get("module", "unknown") for a in alerts)
        rule_counts = Counter(a.get("rule_id") or a.get("code") or a.get("title") or "unknown" for a in alerts)

        entry = {
            "doi": doi,
            "journal_key": jkey,
            "pub_date": eligible_map.get(doi, {}).get("pub_date", ""),
            "title": eligible_map.get(doi, {}).get("title", ""),
            "dir": rec["dir"],
            "n_files": rec["n_files"],
            "files_scanned": scanned_files,
            "runtime_ms": elapsed_ms,
            "error": err,
            "n_alerts": len(alerts),
            "n_critical": sev_counts.get("CRITICAL", 0),
            "n_high": sev_counts.get("HIGH", 0),
            "n_medium": sev_counts.get("MEDIUM", 0),
            "has_crit_or_high": (sev_counts.get("CRITICAL", 0) + sev_counts.get("HIGH", 0)) > 0,
            "modules_triggered": sorted(mod_counts.keys()),
            "rule_counts": dict(rule_counts),
            "alerts": alerts,
        }
        per_paper.append(entry)
        for a in alerts:
            all_alerts.append({
                "doi": doi,
                "journal_key": jkey,
                "severity": a.get("severity", ""),
                "module": a.get("module", ""),
                "rule_id": a.get("rule_id") or a.get("code") or "",
                "file": pathlib.Path(str(a.get("file") or a.get("location") or "")).name,
                "message": str(a.get("message") or a.get("summary") or "")[:300],
            })

        if idx % 35 == 0 or idx == len(targets):
            flagged_any = sum(1 for x in per_paper if x["n_alerts"] > 0)
            flagged_ch = sum(1 for x in per_paper if x["has_crit_or_high"])
            print(f"  [{idx}/{len(targets)}] {time.time()-t0:.1f}s | flagged_any={flagged_any} | flagged_crit_or_high={flagged_ch} | total_alerts={len(all_alerts)}", flush=True)

    # Aggregate statistics
    runtimes = sorted(x["runtime_ms"] for x in per_paper)
    med_rt = runtimes[len(runtimes) // 2] if runtimes else 0.0
    p95_rt = runtimes[int(0.95 * (len(runtimes) - 1))] if runtimes else 0.0

    mod_paper_hits = Counter()
    rule_paper_hits = Counter()
    sev_paper_hits = Counter()
    for x in per_paper:
        for m in x["modules_triggered"]:
            mod_paper_hits[m] += 1
        for r_id in x["rule_counts"]:
            rule_paper_hits[r_id] += 1
        if x["n_critical"] > 0:
            sev_paper_hits["CRITICAL"] += 1
        elif x["n_high"] > 0:
            sev_paper_hits["HIGH_ONLY"] += 1
        elif x["n_medium"] > 0:
            sev_paper_hits["MEDIUM_ONLY"] += 1
        else:
            sev_paper_hits["CLEAN_UNFLAGGED"] += 1

    # Deterministic random sample of 30 unflagged papers (seed=2026) for Negative-Control Manual Audit
    unflagged = [x for x in per_paper if x["n_alerts"] == 0 and not x["error"]]
    unflagged_sorted = sorted(unflagged, key=lambda x: x["doi"])
    rng = random.Random(2026)
    neg_sample = rng.sample(unflagged_sorted, min(30, len(unflagged_sorted)))
    neg_sample_out = [
        {
            "doi": x["doi"],
            "journal_key": x["journal_key"],
            "pub_date": x["pub_date"],
            "dir": x["dir"],
            "n_files": x["n_files"],
            "title": x["title"],
        }
        for x in sorted(neg_sample, key=lambda z: z["doi"])
    ]

    summary = {
        "n_papers_scanned": len(per_paper),
        "n_errors": sum(1 for x in per_paper if x["error"]),
        "total_wall_seconds": round(time.time() - t0, 2),
        "median_runtime_ms": med_rt,
        "p95_runtime_ms": p95_rt,
        "papers_flagged_any": sum(1 for x in per_paper if x["n_alerts"] > 0),
        "papers_flagged_crit_or_high": sum(1 for x in per_paper if x["has_crit_or_high"]),
        "papers_clean_unflagged": len(unflagged),
        "severity_paper_distribution": dict(sev_paper_hits),
        "module_paper_hits": dict(mod_paper_hits),
        "rule_paper_hits": dict(rule_paper_hits.most_common(30)),
        "by_journal": {},
    }
    for jk in sorted(set(x["journal_key"] for x in per_paper)):
        sub = [x for x in per_paper if x["journal_key"] == jk]
        summary["by_journal"][jk] = {
            "scanned": len(sub),
            "flagged_any": sum(1 for x in sub if x["n_alerts"] > 0),
            "flagged_crit_or_high": sum(1 for x in sub if x["has_crit_or_high"]),
            "clean_unflagged": sum(1 for x in sub if x["n_alerts"] == 0),
        }

    # Check our 4 core benchmark cases inside the unbiased scan!
    core_check = {}
    for doi in ("10.1038/s41746-026-02991-x", "10.1038/s41746-026-02470-3", "10.1038/s41467-026-78947-2", "10.1371/journal.pmed.1005078"):
        m = [x for x in per_paper if x["doi"] == doi]
        if m:
            core_check[doi] = {
                "n_alerts": m[0]["n_alerts"],
                "n_critical": m[0]["n_critical"],
                "n_high": m[0]["n_high"],
                "modules": m[0]["modules_triggered"],
                "rules": m[0]["rule_counts"],
                "runtime_ms": m[0]["runtime_ms"],
            }
    summary["core_4_blind_detection"] = core_check

    (PROC_DIR / "clinaudit_2026_scan_results.json").write_text(
        json.dumps({"summary": summary, "papers": per_paper}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PROC_DIR / "negative_control_sample_30.json").write_text(
        json.dumps({"seed": 2026, "population_unflagged": len(unflagged), "sample_size": len(neg_sample_out), "sample": neg_sample_out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(PROC_DIR / "clinaudit_2026_alerts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doi", "journal_key", "severity", "module", "rule_id", "file", "message"])
        w.writeheader()
        for a in all_alerts:
            w.writerow(a)

    print("\n=== CLINAUDIT 2026 COHORT SCAN SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
