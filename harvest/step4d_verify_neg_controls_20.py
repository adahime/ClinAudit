#!/usr/bin/env python3
"""Deep line-by-line & tabular forensic check of the 20 added negative controls (idx 31..50)."""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clinaudit.io_utils import read_xlsx

PROC = ROOT / "harvest" / "processed"
ART = ROOT / "harvest" / "artifacts_n3"
CI_RE = re.compile(r"([0-9]+\.[0-9]+)\s*[\(\[]\s*([0-9]+\.[0-9]+)\s*[-–—,]\s*([0-9]+\.[0-9]+)\s*[\)\]]")


def main():
    d = json.loads((PROC / "negative_control_sample_50_raw.json").read_text(encoding="utf-8"))
    print("=== 1. INSPECT 10.1038/s41746-026-02782-4 (ALL 14 HITS) ===")
    for r in d["added_20_audit"]:
        if r["doi"] == "10.1038/s41746-026-02782-4":
            for h in r["pattern_hits"]:
                print(f"  {h['file']}:{h['line']} | {h['text']}")
            lines = (ART / r["dir"] / "gh_01_train.py").read_text(encoding="utf-8", errors="replace").splitlines()
            print("-- gh_01_train.py L85-115 --")
            for ln in range(85, min(len(lines), 115) + 1):
                print(f"{ln:3d}: {lines[ln-1]}")

    print("\n=== 2. INSPECT ALL 20 NEW CONTROLS TABULAR FILES & INLINE CIs ===")
    anomalies = 0
    for r in d["added_20_audit"]:
        pdir = ART / r["dir"]
        files = sorted(pdir.iterdir()) if pdir.exists() else []
        non_std = [f.name for f in files if f.suffix.lower() not in (".py", ".r", ".ipynb", ".xlsx", ".csv", ".tsv")]
        tab_summary = []
        for f in files:
            if f.suffix.lower() == ".xlsx":
                try:
                    shs = read_xlsx(f)
                    ci_cnt = 0
                    bad_ci = []
                    for sh, rows in shs.items():
                        for ri, row in enumerate(rows[:500]):
                            for cell in row:
                                if isinstance(cell, str):
                                    for m in CI_RE.finditer(cell):
                                        ci_cnt += 1
                                        pt, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
                                        if hi < lo or (lo == hi and 0.05 < pt < 0.995):
                                            bad_ci.append((sh, ri, cell[:50]))
                    tab_summary.append(f"{f.name}({len(shs)}sh,{ci_cnt}CI,{len(bad_ci)}bad)")
                    if bad_ci:
                        anomalies += len(bad_ci)
                        print(f"    BAD CI in {r['doi']} {f.name}: {bad_ci[:3]}")
                except Exception as e:
                    tab_summary.append(f"{f.name}(ERR:{e})")
        print(f"  [{r['idx']:2d}] {r['doi']} | tabs={tab_summary[:3]} | other={non_std[:3]}")
    print(f"\nTotal tabular CI anomalies across 20 added controls: {anomalies}")


if __name__ == "__main__":
    main()
