#!/usr/bin/env python3
"""
Step 4B: Run calibrated `clinaudit v0.2.0` across ALL 174 downloaded 2026 clinical ML papers
(preserving `clinaudit_2026_scan_results.json` untouched as the frozen v0.1.0 blind baseline).
"""
import csv
import json
import pathlib
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinaudit import __version__
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
    print(f"Scanning {len(targets)} downloaded 2026 papers with ClinAudit v{__version__}...", flush=True)

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
        rule_counts = Counter(a.get("rule_id") or a.get("code") or "unknown" for a in alerts)

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
                "rule_id": a.get("rule_id") or "",
                "file": pathlib.Path(str(a.get("file") or "")).name,
                "message": str(a.get("message") or "")[:300],
            })
        if idx % 35 == 0 or idx == len(targets):
            flagged_any = sum(1 for x in per_paper if x["n_alerts"] > 0)
            print(f"  [{idx}/{len(targets)}] {time.time()-t0:.1f}s | flagged={flagged_any} | alerts={len(all_alerts)}", flush=True)

    runtimes = sorted(x["runtime_ms"] for x in per_paper)
    med_rt = runtimes[len(runtimes) // 2] if runtimes else 0.0
    p95_rt = runtimes[int(0.95 * (len(runtimes) - 1))] if runtimes else 0.0

    flagged_list = [
        {
            "doi": x["doi"],
            "journal_key": x["journal_key"],
            "n_alerts": x["n_alerts"],
            "rules": sorted(x["rule_counts"].keys()),
            "modules": x["modules_triggered"],
            "title": x["title"][:90],
        }
        for x in per_paper if x["n_alerts"] > 0
    ]

    # Check all 7 confirmed defects!
    known_7 = [
        "10.1038/s41746-026-02991-x",
        "10.1038/s41746-026-02470-3",
        "10.1038/s41467-026-78947-2",
        "10.1371/journal.pmed.1005078",
        "10.1038/s41467-026-70584-z",
        "10.1038/s41467-026-74209-3",
        "10.1371/journal.pmed.1005190",
    ]
    det_7 = {}
    for doi in known_7:
        m = [x for x in per_paper if x["doi"] == doi]
        det_7[doi] = {"detected": bool(m and m[0]["n_alerts"] > 0), "n_alerts": m[0]["n_alerts"] if m else 0, "rules": sorted(m[0]["rule_counts"].keys()) if m else []}

    summary = {
        "version": __version__,
        "n_papers_scanned": len(per_paper),
        "n_errors": sum(1 for x in per_paper if x["error"]),
        "total_wall_seconds": round(time.time() - t0, 2),
        "median_runtime_ms": med_rt,
        "p95_runtime_ms": p95_rt,
        "papers_flagged": len(flagged_list),
        "papers_clean": len(per_paper) - len(flagged_list),
        "confirmed_7_detection": det_7,
        "confirmed_7_recall_n": sum(1 for v in det_7.values() if v["detected"]),
        "flagged_papers": flagged_list,
    }
    (PROC_DIR / "clinaudit_2026_scan_v020.json").write_text(
        json.dumps({"summary": summary, "papers": per_paper}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(PROC_DIR / "clinaudit_2026_alerts_v020.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doi", "journal_key", "severity", "module", "rule_id", "file", "message"])
        w.writeheader()
        for a in all_alerts:
            w.writerow(a)
    print("\n=== CLINAUDIT v0.2.0 2026 COHORT SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
