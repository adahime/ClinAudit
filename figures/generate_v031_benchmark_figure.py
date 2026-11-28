#!/usr/bin/env python3
"""Generate Figure 5: ClinAudit v0.2.0 -> v0.3.0 -> v0.3.1 Performance & Time-Cost Benchmark (SVG + 3x PNG)."""

from __future__ import annotations

import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# Okabe-Ito colorblind-safe palette (matching Figures 1-4)
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_SLATE = "#334155"
C_DARK = "#0F172A"
C_MUTED = "#475569"
C_LIGHT = "#E2E8F0"
C_PANEL = "#F8FAFC"


def render_svg_to_png(svg_path: pathlib.Path, png_path: pathlib.Path, width: int, height: int) -> None:
    html_path = svg_path.with_suffix(".html")
    svg_text = svg_path.read_text(encoding="utf-8")
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        f"* {{ margin:0; padding:0; box-sizing:border-box; }}"
        f"html, body {{ width:{width}px; height:{height}px; overflow:hidden; background:#ffffff; }}"
        f"svg {{ display:block; width:{width}px; height:{height}px; }}"
        f"</style></head><body>{svg_text}</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    file_url = "file:///" + str(html_path).replace("\\", "/")
    cmd = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--force-device-scale-factor=3",
        f"--screenshot={png_path}",
        file_url,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    html_path.unlink(missing_ok=True)
    print(f"Rendered {png_path.name} ({png_path.stat().st_size:,} B, {width * 3}x{height * 3} px @ 3x)")


def build_svg() -> tuple[str, int, int]:
    W, H = 1240, 490
    parts: list[str] = []
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

    # Header
    parts.append(
        '<text x="28" y="34" font-size="18" font-weight="700" fill="#0F172A">'
        'ClinAudit Engine Evolution: v0.2.0 &#x2192; v0.3.0 &#x2192; v0.3.1 Empirical Benchmark</text>'
    )
    parts.append(
        '<text x="28" y="54" font-size="11.5" fill="#475569">'
        'Zero-regression recall lock across 51 verified publisher .xlsx defects (100.0%) &#xB7; '
        '338-paper held-out validation &#xB7; 293-paper out-of-sample residual stress test &#xB7; 0 crashes / 0 timeouts</text>'
    )
    parts.append(f'<line x1="28" y1="66" x2="{W - 28}" y2="66" stroke="{C_LIGHT}" stroke-width="1.2"/>')

    # =========================================================================
    # PANEL A: Held-Out FP Reduction & P0 Precision (Wave 5, N=338 papers)
    # =========================================================================
    ax, ay, aw, ah = 28, 82, 378, 354
    parts.append(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" rx="8" fill="{C_PANEL}" stroke="{C_LIGHT}" stroke-width="1"/>')
    parts.append(f'<text x="{ax + 14}" y="{ay + 24}" font-size="15" font-weight="700" fill="{C_DARK}">a</text>')
    parts.append(f'<text x="{ax + 30}" y="{ay + 24}" font-size="13" font-weight="700" fill="{C_DARK}">Held-Out False Positives &amp; PPV</text>')
    parts.append(f'<text x="{ax + 30}" y="{ay + 41}" font-size="10.5" fill="{C_MUTED}">Closed-code .xlsx cohort (N = 338 papers; 7 verified)</text>')

    # Recall badge
    parts.append(f'<rect x="{ax + aw - 122}" y="{ay + 10}" width="110" height="22" rx="11" fill="#DCFCE7" stroke="#86EFAC" stroke-width="1"/>')
    parts.append(f'<text x="{ax + aw - 67}" y="{ay + 25}" text-anchor="middle" font-size="10" font-weight="700" fill="#166534">Recall: 7/7 (100%)</text>')

    # Vertical bars inside Panel A
    base_y = ay + 252
    top_y = ay + 82
    plot_h = base_y - top_y
    max_fp = 70.0
    # Y gridlines
    for gv in (0, 20, 40, 60):
        gy = base_y - (gv / max_fp) * plot_h
        parts.append(f'<line x1="{ax + 48}" y1="{gy:.1f}" x2="{ax + aw - 18}" y2="{gy:.1f}" stroke="#CBD5E1" stroke-dasharray="3,3" stroke-width="0.9"/>')
        parts.append(f'<text x="{ax + 42}" y="{gy + 4:.1f}" text-anchor="end" font-size="10" fill="{C_MUTED}">{gv}</text>')
    parts.append(f'<line x1="{ax + 48}" y1="{base_y}" x2="{ax + aw - 18}" y2="{base_y}" stroke="{C_SLATE}" stroke-width="1.2"/>')
    parts.append(f'<text x="{ax + 16}" y="{(top_y + base_y) / 2:.1f}" transform="rotate(-90 {ax + 16} {(top_y + base_y) / 2:.1f})" text-anchor="middle" font-size="10.5" font-weight="600" fill="{C_SLATE}">False-Positive Papers</text>')

    bars_a = [
        ("v0.2.0", "All alerts", 61, C_VERM, "61 FPs", "PPV 10.3%", "FPR 18.4%"),
        ("v0.3.0 / 0.3.1", "P0 + P1", 11, C_BLUE, "11 (-82%)", "PPV 38.9%", "FPR 3.3%"),
        ("v0.3.0 / 0.3.1", "P0 only", 3, C_GREEN, "3 (-95.1%)", "PPV 70.0%", "FPR 0.91%"),
    ]
    bw = 74
    xs_a = [ax + 72, ax + 178, ax + 284]
    for (lbl1, lbl2, val, col, top_txt, ppv_txt, fpr_txt), bx in zip(bars_a, xs_a):
        bh = (val / max_fp) * plot_h
        by = base_y - bh
        parts.append(f'<rect x="{bx}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" rx="4" fill="{col}"/>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="10.5" font-weight="700" fill="{col}">{top_txt}</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{base_y + 16}" text-anchor="middle" font-size="10.5" font-weight="700" fill="{C_DARK}">{lbl1}</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{base_y + 30}" text-anchor="middle" font-size="9.5" fill="{C_MUTED}">{lbl2}</text>')
        parts.append(f'<rect x="{bx - 4}" y="{base_y + 39}" width="{bw + 8}" height="34" rx="5" fill="#FFFFFF" stroke="{C_LIGHT}" stroke-width="1"/>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{base_y + 53}" text-anchor="middle" font-size="10" font-weight="700" fill="{col}">{ppv_txt}</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{base_y + 66}" text-anchor="middle" font-size="9.5" fill="{C_MUTED}">{fpr_txt}</text>')

    # Callout arrow 6.8x
    parts.append(f'<rect x="{ax + 150}" y="{ay + 74}" width="198" height="24" rx="5" fill="#ECFDF5" stroke="#6EE7B7" stroke-width="1"/>')
    parts.append(f'<text x="{ax + 249}" y="{ay + 90}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#065F46">P0 Precision: 10.3% &#x2192; 70.0% (6.8&#xD7;)</text>')

    # =========================================================================
    # PANEL B: Out-of-Sample Residual P0 Noise (v0.3.0 -> v0.3.1) & 51-Hit Lock
    # =========================================================================
    bx0, by0, bw0, bh0 = 424, 82, 386, 354
    parts.append(f'<rect x="{bx0}" y="{by0}" width="{bw0}" height="{bh0}" rx="8" fill="{C_PANEL}" stroke="{C_LIGHT}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0 + 14}" y="{by0 + 24}" font-size="15" font-weight="700" fill="{C_DARK}">b</text>')
    parts.append(f'<text x="{bx0 + 30}" y="{by0 + 24}" font-size="13" font-weight="700" fill="{C_DARK}">Out-of-Sample Tail Cut &amp; Recall Lock</text>')
    parts.append(f'<text x="{bx0 + 30}" y="{by0 + 41}" font-size="10.5" fill="{C_MUTED}">293 unseen Wave 4 .xlsx papers (10 new defects found)</text>')

    # Horizontal stacked bar comparison: v0.3.0 vs v0.3.1 P0 candidates on Set C (293 papers)
    bar_x = bx0 + 88
    bar_max_w = 248
    max_cand = 60.0
    rows_b = [
        ("v0.3.0 P0", 10, 2, 42, by0 + 84, "54 P0 (PPV 18.5%)"),
        ("v0.3.1 P0", 10, 2, 19, by0 + 142, "31 P0 (PPV 32.3%)"),
    ]
    parts.append(f'<text x="{bx0 + 16}" y="{by0 + 68}" font-size="10.5" font-weight="700" fill="{C_SLATE}">Unseen Wave 4 Tail: P0 Alert Composition</text>')
    for rlbl, n_tp, n_bord, n_fp, ry, rnote in rows_b:
        w_tp = (n_tp / max_cand) * bar_max_w
        w_bd = (n_bord / max_cand) * bar_max_w
        w_fp = (n_fp / max_cand) * bar_max_w
        parts.append(f'<text x="{bar_x - 8}" y="{ry + 17}" text-anchor="end" font-size="11" font-weight="700" fill="{C_DARK}">{rlbl}</text>')
        parts.append(f'<rect x="{bar_x}" y="{ry}" width="{w_tp:.1f}" height="26" rx="3" fill="{C_GREEN}"/>')
        parts.append(f'<rect x="{bar_x + w_tp:.1f}" y="{ry}" width="{w_bd:.1f}" height="26" fill="{C_ORANGE}"/>')
        parts.append(f'<rect x="{bar_x + w_tp + w_bd:.1f}" y="{ry}" width="{w_fp:.1f}" height="26" rx="3" fill="#FCA5A5" stroke="{C_VERM}" stroke-width="0.8"/>')
        parts.append(f'<text x="{bar_x + w_tp / 2:.1f}" y="{ry + 17}" text-anchor="middle" font-size="10" font-weight="700" fill="#FFFFFF">{n_tp}</text>')
        parts.append(f'<text x="{bar_x + w_tp + w_bd + w_fp / 2:.1f}" y="{ry + 17}" text-anchor="middle" font-size="10" font-weight="700" fill="#991B1B">{n_fp} FPs</text>')
        parts.append(f'<text x="{bar_x}" y="{ry + 41}" font-size="10" font-weight="600" fill="{C_MUTED}">{rnote}</text>')

    # Legend for stacked bars
    ly = by0 + 200
    parts.append(f'<rect x="{bx0 + 18}" y="{ly}" width="11" height="11" rx="2" fill="{C_GREEN}"/>')
    parts.append(f'<text x="{bx0 + 34}" y="{ly + 9.5}" font-size="9.8" fill="{C_SLATE}">New Confirmed (10)</text>')
    parts.append(f'<rect x="{bx0 + 144}" y="{ly}" width="11" height="11" rx="2" fill="{C_ORANGE}"/>')
    parts.append(f'<text x="{bx0 + 160}" y="{ly + 9.5}" font-size="9.8" fill="{C_SLATE}">Borderline (2)</text>')
    parts.append(f'<rect x="{bx0 + 242}" y="{ly}" width="11" height="11" rx="2" fill="#FCA5A5" stroke="{C_VERM}" stroke-width="0.8"/>')
    parts.append(f'<text x="{bx0 + 258}" y="{ly + 9.5}" font-size="9.8" fill="{C_SLATE}">Residual P0 FP (-54.8%)</text>')

    # 3 KPI lock cards inside Panel B bottom
    cards = [
        ("51 / 51", "100.0% Recall Lock", "34 W4 + 7 W5 + 10 W4b\n0 misses · 0 downgrades", C_GREEN, "#ECFDF5", "#6EE7B7"),
        ("-54.8%", "Extra P0 FP Cut", "v0.3.0 (42) &#x2192; v0.3.1 (19)\n4 surgical parser guards", C_BLUE, "#EFF6FF", "#93C5FD"),
        ("0 / 409", "Timeouts &amp; Crashes", "v0.2.0 had 22 (&gt;45s)\nO(N&#xB2;) stall eliminated", C_SLATE, "#F1F5F9", "#CBD5E1"),
    ]
    cw = 114
    for idx, (big, sub1, sub2, col, bgc, bdc) in enumerate(cards):
        cx = bx0 + 14 + idx * (cw + 8)
        cy = by0 + 228
        parts.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="110" rx="6" fill="{bgc}" stroke="{bdc}" stroke-width="1"/>')
        parts.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + 30}" text-anchor="middle" font-size="18" font-weight="700" fill="{col}">{big}</text>')
        parts.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + 49}" text-anchor="middle" font-size="10" font-weight="700" fill="{C_DARK}">{sub1}</text>')
        for li, line in enumerate(sub2.split("\n")):
            parts.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + 70 + li * 15}" text-anchor="middle" font-size="9" fill="{C_MUTED}">{line}</text>')

    # =========================================================================
    # PANEL C: Measured P0 Adjudication Queue Across Both Real Cohorts (100% Measured)
    # =========================================================================
    cx0, cy0, cw0, ch0 = 828, 82, 384, 354
    parts.append(f'<rect x="{cx0}" y="{cy0}" width="{cw0}" height="{ch0}" rx="8" fill="{C_PANEL}" stroke="{C_LIGHT}" stroke-width="1"/>')
    parts.append(f'<text x="{cx0 + 14}" y="{cy0 + 24}" font-size="15" font-weight="700" fill="{C_DARK}">c</text>')
    parts.append(f'<text x="{cx0 + 30}" y="{cy0 + 24}" font-size="13" font-weight="700" fill="{C_DARK}">Measured Adjudication Queue Shrinkage</text>')
    parts.append(f'<text x="{cx0 + 30}" y="{cy0 + 41}" font-size="10.5" fill="{C_MUTED}">Exact papers requiring human/LLM review (0 estimates)</text>')

    # Group 1: Wave 4 Full Cohort (N=1,542 .xlsx papers) — 328 -> 83 -> 60
    px_l = cx0 + 82
    pw_c = 224.0
    parts.append(f'<text x="{cx0 + 16}" y="{cy0 + 66}" font-size="10.5" font-weight="700" fill="{C_SLATE}">Wave 4 Full Cohort (N = 1,542 .xlsx papers; 44 verified)</text>')
    w4_bars = [
        ("v0.2.0", 328, 340.0, C_VERM, "328 papers (10.4% PPV)"),
        ("v0.3.0 P0", 83, 340.0, C_BLUE, "83 (-74.7% · 47.0% PPV)"),
        ("v0.3.1 P0", 60, 340.0, C_GREEN, "60 (-81.7% · 65.0% PPV)"),
    ]
    for i, (lbl, val, vmax, col, note) in enumerate(w4_bars):
        ry = cy0 + 76 + i * 28
        bw_val = max(8.0, (val / vmax) * pw_c)
        parts.append(f'<text x="{px_l - 8}" y="{ry + 14}" text-anchor="end" font-size="10" font-weight="700" fill="{C_DARK}">{lbl}</text>')
        parts.append(f'<rect x="{px_l}" y="{ry}" width="{bw_val:.1f}" height="20" rx="3.5" fill="{col}"/>')
        if bw_val > 145:
            parts.append(f'<text x="{px_l + bw_val - 6:.1f}" y="{ry + 14}" text-anchor="end" font-size="9.5" font-weight="700" fill="#FFFFFF">{note}</text>')
        else:
            parts.append(f'<text x="{px_l + bw_val + 6:.1f}" y="{ry + 14}" font-size="9.5" font-weight="700" fill="{col}">{note}</text>')

    # Group 2: Wave 5 Closed-Code Cohort (N=338 .xlsx papers) — 68 -> 18 -> 10
    parts.append(f'<text x="{cx0 + 16}" y="{cy0 + 178}" font-size="10.5" font-weight="700" fill="{C_SLATE}">Wave 5 Held-Out Cohort (N = 338 .xlsx papers; 7 verified)</text>')
    w5_bars = [
        ("v0.2.0", 68, 80.0, C_VERM, "68 papers (10.3% PPV)"),
        ("v0.3.x P0+P1", 18, 80.0, C_BLUE, "18 (-73.5% · 38.9% PPV)"),
        ("v0.3.x P0", 10, 80.0, C_GREEN, "10 (-85.3% · 70.0% PPV)"),
    ]
    for i, (lbl, val, vmax, col, note) in enumerate(w5_bars):
        ry = cy0 + 188 + i * 28
        bw_val = max(8.0, (val / vmax) * pw_c)
        parts.append(f'<text x="{px_l - 8}" y="{ry + 14}" text-anchor="end" font-size="10" font-weight="700" fill="{C_DARK}">{lbl}</text>')
        parts.append(f'<rect x="{px_l}" y="{ry}" width="{bw_val:.1f}" height="20" rx="3.5" fill="{col}"/>')
        if bw_val > 145:
            parts.append(f'<text x="{px_l + bw_val - 6:.1f}" y="{ry + 14}" text-anchor="end" font-size="9.5" font-weight="700" fill="#FFFFFF">{note}</text>')
        else:
            parts.append(f'<text x="{px_l + bw_val + 6:.1f}" y="{ry + 14}" font-size="9.5" font-weight="700" fill="{col}">{note}</text>')

    # Bottom callout box inside Panel C (epistemic advantage over blind LLM review)
    parts.append(f'<rect x="{cx0 + 14}" y="{cy0 + 278}" width="{cw0 - 28}" height="64" rx="6" fill="#FFF7ED" stroke="#FDBA74" stroke-width="1"/>')
    parts.append(
        f'<text x="{cx0 + 23}" y="{cy0 + 295}" font-size="10" font-weight="700" fill="#9A3412">'
        'Why Not a Hypothetical "Blind LLM" Bar?</text>'
    )
    parts.append(
        f'<text x="{cx0 + 23}" y="{cy0 + 310}" font-size="9.2" fill="#7C2D12">'
        'Blind LLM review cannot exhaustively scan massive multi-file .xlsx workbooks</text>'
    )
    parts.append(
        f'<text x="{cx0 + 23}" y="{cy0 + 323}" font-size="9.2" fill="#7C2D12">'
        'or spontaneously test hidden axioms (mid-table row shifts, tail p &gt; 1, sqrt(B)).</text>'
    )
    parts.append(
        f'<text x="{cx0 + 23}" y="{cy0 + 336}" font-size="9.2" font-weight="600" fill="#9A3412">'
        'ClinAudit deterministically flags exact (sheet, row, col) P0 coordinates.</text>'
    )

    # Footer bar
    parts.append(
        f'<text x="28" y="{H - 22}" font-size="10.2" fill="{C_MUTED}">'
        'Deterministic 400-case synthetic benchmark: 99.0% sensitivity, 100.0% specificity, F1 = 0.995 (median 0.81 ms/case) &#xB7; '
        '27/27 unit tests pass &#xB7; Zero-heavy-dependency (Python stdlib + NumPy) &#xB7; Immutable v0.2.0 Zenodo tag preserved.</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        '<style>text { font-family: Arial, Helvetica, sans-serif; }</style>\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    return svg, W, H


def main() -> None:
    svg, w, h = build_svg()
    svg_path = FIG_DIR / "Figure5_v020_to_v031_Performance.svg"
    png_path = FIG_DIR / "Figure5_v020_to_v031_Performance.png"
    svg_path.write_text(svg, encoding="utf-8")
    render_svg_to_png(svg_path, png_path, w, h)


if __name__ == "__main__":
    main()
