#!/usr/bin/env python3
"""
Standalone Clinical Safety Verification Script for PLOS Medicine (2026-10-06):
  Zhu et al., DOI: 10.1371/journal.pmed.1005078 ("CKMAI" & "CKMAIcalc")

Dependencies: Python stdlib + NumPy ONLY.
Does NOT redistribute any copyrighted third-party files; downloads official public
artifacts directly from PLOS Medicine, Zenodo, and GitHub into `--cache-dir`,
verifies exact SHA-256 hashes, and writes derived verification summaries.
"""
import argparse
import csv
import gzip
import hashlib
import json
import pathlib
import struct
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

OFFICIAL_ARTIFACTS = {
    "S1_Data.xlsx": {
        "url": "https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1005078.s007&type=supplementary",
        "sha256": "ae6248050bb6143ed09dd7c58f1f9e045ccb806e16190071ff4f3889bed1aaeb",
    },
    "data_HOS.xlsx": {
        "url": "https://zenodo.org/api/records/22668125/files/data_HOS.xlsx/content",
        "sha256": "7beb0ab9868d8bdb96e0399cd0dd3cd2d58c2a6a6c027fd485d8b67c8ba9e889",
    },
    "CKMAI_paper_code.zip": {
        "url": "https://github.com/zhuzhengyang517/CKMAI_paper_code/archive/refs/heads/master.zip",
        "sha256": "ec9da40d61c89b1ad1145a99b6150ce9c5742218fce2260422cdf5243f82c74c",
    },
    "CKMAIcalc.zip": {
        "url": "https://github.com/zhuzhengyang517/CKMAIcalc/archive/refs/heads/master.zip",
        "sha256": "084549029fd7cace024dd502cfadc6f8c2ec347351ed8ff4d00bbd3a96e86ecc",
    },
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_artifacts(cache_dir: pathlib.Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for fname, meta in OFFICIAL_ARTIFACTS.items():
        p = cache_dir / fname
        if not p.exists():
            req = urllib.request.Request(meta["url"], headers={"User-Agent": "ClinAuditSafetyVerifier/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                p.write_bytes(r.read())
        h = sha256_file(p)
        hashes[fname] = {
            "bytes": p.stat().st_size,
            "sha256": h,
            "expected_sha256": meta["sha256"],
            "exact_match": (h == meta["sha256"]),
        }
    return hashes


def read_xlsx_sheet(path: pathlib.Path, sheet_idx: int = 1):
    with zipfile.ZipFile(path) as zf:
        sst = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("m:si", ns):
                sst.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        sroot = ET.fromstring(zf.read(f"xl/worksheets/sheet{sheet_idx}.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = []
        for row_el in sroot.findall(".//m:sheetData/m:row", ns):
            rd, mc = {}, -1
            for c in row_el.findall("m:c", ns):
                ref = c.attrib.get("r", "")
                cl = "".join(ch for ch in ref if ch.isalpha())
                ci = 0
                for ch in cl:
                    ci = ci * 26 + (ord(ch.upper()) - ord("A") + 1)
                ci -= 1
                mc = max(mc, ci)
                t = c.attrib.get("t", "")
                v = c.find("m:v", ns)
                if t == "s" and v is not None and v.text is not None:
                    val = sst[int(v.text)]
                elif v is not None and v.text is not None:
                    val = v.text
                else:
                    val = ""
                rd[ci] = val
            if mc >= 0:
                rows.append([rd.get(i, "") for i in range(mc + 1)])
        return rows


class RDSParser:
    def __init__(self, b: bytes):
        self.b = b
        self.pos = 0
        self.ref_table = []

    def read(self, n):
        s = self.b[self.pos : self.pos + n]
        self.pos += n
        return s

    def read_int(self):
        return struct.unpack(">i", self.read(4))[0]

    def read_uint(self):
        return struct.unpack(">I", self.read(4))[0]

    def parse(self):
        self.read(2)
        fmt_ver = self.read_int()
        self.read_int()
        self.read_int()
        if fmt_ver == 3:
            enc_len = self.read_int()
            self.read(enc_len)
        return self.read_item()

    def read_item(self):
        flags = self.read_uint()
        t = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if t in (0, 254, 253, 252, 251, 241, 242, 244, 245, 246, 247, 21):
            return None
        if t == 255:
            idx = flags >> 8
            if idx == 0:
                idx = self.read_int()
            return self.ref_table[idx - 1] if 0 < idx <= len(self.ref_table) else None
        if t == 1:
            sym = self.read_item()
            self.ref_table.append(sym)
            return sym
        if t == 9:
            n = self.read_int()
            return None if n == -1 else self.read(n).decode("utf-8", errors="replace")
        if t == 4:
            self.read_int()
            env = {"type": "env"}
            self.ref_table.append(env)
            for _ in range(4):
                self.read_item()
            return env
        if t == 3:
            if has_attr:
                self.read_item()
            if has_tag:
                self.read_item()
            self.read_item()
            self.read_item()
            return {"type": "closure"}
        if t in (2, 6):
            if has_attr:
                self.read_item()
            tag = self.read_item() if has_tag else None
            car = self.read_item()
            cdr = self.read_item()
            res = [(tag, car)]
            if isinstance(cdr, list):
                res.extend(cdr)
            elif cdr is not None:
                res.append((None, cdr))
            return res
        if t in (10, 13):
            n = self.read_int()
            vals = list(struct.unpack(f">{n}i", self.read(4 * n))) if n > 0 else []
            attrs = self.read_item() if has_attr else None
            return {"type": "int", "val": vals, "attrs": attrs}
        if t == 14:
            n = self.read_int()
            vals = list(struct.unpack(f">{n}d", self.read(8 * n))) if n > 0 else []
            attrs = self.read_item() if has_attr else None
            return {"type": "real", "val": vals, "attrs": attrs}
        if t == 16:
            n = self.read_int()
            vals = [self.read_item() for _ in range(n)]
            attrs = self.read_item() if has_attr else None
            return {"type": "str", "val": vals, "attrs": attrs}
        if t in (19, 20):
            n = self.read_int()
            vals = [self.read_item() for _ in range(n)]
            attrs = self.read_item() if has_attr else None
            return {"type": "list", "val": vals, "attrs": attrs}
        if t == 24:
            n = self.read_int()
            self.read(n)
            if has_attr:
                self.read_item()
            return {"type": "raw"}
        if t == 238:
            for _ in range(3):
                self.read_item()
            return {"type": "altrep"}
        if t in (248, 249, 250):
            nm = self.read_item()
            self.ref_table.append(nm)
            return nm
        if t in (7, 8):
            n = self.read_int()
            return self.read(n).decode("utf-8", errors="replace")
        raise ValueError(f"Unsupported SEXP {t}")


def rlist_to_dict(obj):
    names = [v for k, v in (obj.get("attrs") or []) if k == "names"][0]["val"]
    return dict(zip(names, obj["val"]))


def fit_cox_breslow(X_mat, t, e):
    order = np.argsort(-t, kind="mergesort")
    Xs, ts, es = X_mat[order], t[order], e[order]
    n_obs, p = Xs.shape
    idx_end = np.arange(n_obs)
    for k in range(n_obs - 2, -1, -1):
        if ts[k] == ts[k + 1]:
            idx_end[k] = idx_end[k + 1]
    ev_idx = np.where(es == 1)[0]
    mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xs - mu) / sd
    beta = np.zeros(p)

    def eval_ll(b):
        eta = Z @ b
        eta -= np.max(eta)
        w = np.exp(eta)
        S0 = np.cumsum(w)[idx_end]
        S1 = np.cumsum(Z * w[:, None], axis=0)[idx_end]
        ll = float(np.sum(eta[ev_idx] - np.log(S0[ev_idx])))
        mx = S1[ev_idx] / S0[ev_idx, None]
        grad = np.sum(Z[ev_idx] - mx, axis=0)
        outer = (Z[:, :, None] * Z[:, None, :]) * w[:, None, None]
        S2 = np.cumsum(outer, axis=0)[idx_end]
        I = np.sum(S2[ev_idx] / S0[ev_idx, None, None], axis=0) - (mx.T @ mx)
        return ll, grad, I

    ll, grad, I = eval_ll(beta)
    for _ in range(50):
        step = np.linalg.solve(I + 1e-9 * np.eye(p), grad)
        a = 1.0
        while a > 1e-6:
            lln, gn, In = eval_ll(beta + a * step)
            if lln >= ll - 1e-12:
                beta, ll, grad, I = beta + a * step, lln, gn, In
                break
            a *= 0.5
        if np.max(np.abs(a * step)) < 1e-10:
            break
    se_z = np.sqrt(np.diag(np.linalg.inv(I)))
    return beta / sd, se_z / sd, ll


def harrell_c(t, e, score):
    ev_idx = np.where(e == 1)[0]
    conc, perm = 0.0, 0.0
    for i in ev_idx:
        ti, si = t[i], score[i]
        mask = (t > ti) | ((t == ti) & (e == 0))
        sj = score[mask]
        perm += len(sj)
        conc += np.sum(si > sj) + 0.5 * np.sum(si == sj)
    return float(conc / perm)


def roc_auc(y, score):
    p, q = score[y == 1], score[y == 0]
    return float(np.mean((p[:, None] > q[None, :]) + 0.5 * (p[:, None] == q[None, :])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(pathlib.Path(__file__).resolve().parent / "_official_cache_do_not_redistribute"))
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    cache_dir = pathlib.Path(args.cache_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hashes = ensure_artifacts(cache_dir)
    with zipfile.ZipFile(cache_dir / "CKMAIcalc.zip") as zf:
        params = rlist_to_dict(RDSParser(gzip.decompress(zf.read("CKMAIcalc-master/inst/extdata/CKMAI_params.rds"))).parse())
        cph_all = rlist_to_dict(RDSParser(gzip.decompress(zf.read("CKMAIcalc-master/inst/extdata/CKMAI_nomogram_model.rds"))).parse())
        cph_cvd = rlist_to_dict(RDSParser(gzip.decompress(zf.read("CKMAIcalc-master/inst/extdata/CKMAI_nomogram_model_cvd.rds"))).parse())
        glm_dec = gzip.decompress(zf.read("CKMAIcalc-master/inst/extdata/CKM_high_logistic.rds"))
        glm_coefs = np.array(struct.unpack(">9d", glm_dec[39 : 39 + 72]))

    with zipfile.ZipFile(cache_dir / "CKMAI_paper_code.zip") as zf:
        code_lines = zf.read("CKMAI_paper_code-master/code.R").decode("utf-8").splitlines()
        line_648 = code_lines[647].strip()

    rows = read_xlsx_sheet(cache_dir / "S1_Data.xlsx", 1)
    hdr, data = rows[0], rows[1:]
    n = len(data)
    col = lambda nm: np.array([float(r[hdr.index(nm)]) for r in data], dtype=float)

    features = params["features"]["val"]
    weights_abs = np.array(params["weights"]["val"], dtype=float)
    means_rds = np.array(params["mean"]["val"], dtype=float)
    sds_rds = np.array(params["sd"]["val"], dtype=float)
    train_min = float(params["train_min"]["val"][0])
    train_max = float(params["train_max"]["val"][0])

    X = np.zeros((n, len(features)))
    for j, f in enumerate(features):
        if f in hdr:
            X[:, j] = col(f)
        elif f == "Sex1":
            X[:, j] = (col("Sex") == 1).astype(float)
        elif f == "Smoking1":
            X[:, j] = (col("Smoking") == 1).astype(float)
        elif f == "Education3":
            X[:, j] = (col("Education") == 3).astype(float)
        elif f == "Ethnicity2":
            X[:, j] = (col("Ethnicity") == 2).astype(float)
        elif f == "Marital_Status2":
            X[:, j] = (col("Marital_Status") == 2).astype(float)
        elif f == "PIR3":
            X[:, j] = (col("PIR") == 3).astype(float)
        elif f == "Diabetes1":
            X[:, j] = (col("Diabetes") == 1).astype(float)

    Z = (X - means_rds) / sds_rds
    ckmai_pub = 100.0 * ((Z @ weights_abs) - train_min) / (train_max - train_min)
    time_v, mort, cvd = col("Follow_up"), col("Mortality_Status"), col("Death_CVD")
    ckm_high = (col("CKM") >= 3).astype(float)
    age, sex = col("Age"), col("Sex")

    beta_z, se_z, _ = fit_cox_breslow(Z, time_v, mort)
    ckmai_signed = Z @ (weights_abs * np.sign(beta_z))

    # 5-fold OOF CV
    rng = np.random.RandomState(42)
    folds = np.array_split(rng.permutation(n), 5)
    oof_sgn = np.zeros(n)
    for fi in range(5):
        te = folds[fi]
        tr = np.concatenate([folds[k] for k in range(5) if k != fi])
        mu_tr, sd_tr = X[tr].mean(0), X[tr].std(0, ddof=1)
        b_tr, _, _ = fit_cox_breslow((X[tr] - mu_tr) / sd_tr, time_v[tr], mort[tr])
        oof_sgn[te] = ((X[te] - mu_tr) / sd_tr) @ (b_tr / np.max(np.abs(b_tr)))

    # Age+Sex adjusted LRT
    _, _, ll_base = fit_cox_breslow(np.column_stack([age, (sex == 1).astype(float)]), time_v, mort)
    _, _, ll_pub = fit_cox_breslow(np.column_stack([ckmai_pub, age, (sex == 1).astype(float)]), time_v, mort)
    sc_s100 = 100.0 * (ckmai_signed - ckmai_signed.min()) / (ckmai_signed.max() - ckmai_signed.min())
    _, _, ll_sgn = fit_cox_breslow(np.column_stack([sc_s100, age, (sex == 1).astype(float)]), time_v, mort)
    lrt_pub = float(2.0 * (ll_pub - ll_base))
    lrt_sgn = float(2.0 * (ll_sgn - ll_base))

    # CKMAIcalc predict_risks() Sex-encoding proof
    cnames_all = [v for k, v in cph_all["coefficients"]["attrs"] if k == "names"][0]["val"]
    coefs_all = np.array(cph_all["coefficients"]["val"], dtype=float)
    center_all = float(cph_all["center"]["val"][0])
    t_all = np.array(cph_all["time"]["val"])
    s0_all = float(np.array(cph_all["surv"]["val"])[np.where(t_all <= 120.0)[0][-1]])

    cnames_cvd = [v for k, v in cph_cvd["coefficients"]["attrs"] if k == "names"][0]["val"]
    coefs_cvd = np.array(cph_cvd["coefficients"]["val"], dtype=float)
    center_cvd = float(cph_cvd["center"]["val"][0])
    t_cvd = np.array(cph_cvd["time"]["val"])
    s0_cvd = float(np.array(cph_cvd["surv"]["val"])[np.where(t_cvd <= 120.0)[0][-1]])

    X_raw = np.column_stack([col(v) for v in cnames_all])
    X_raw_01 = X_raw.copy()
    X_raw_01[:, cnames_all.index("Sex")] = (sex == 1).astype(float)

    risk_all = 1.0 - (s0_all ** np.exp(X_raw @ coefs_all - center_all))
    risk_cvd_bug = 1.0 - (s0_cvd ** np.exp(X_raw @ coefs_cvd - center_cvd))
    risk_cvd_fix = 1.0 - (s0_cvd ** np.exp(X_raw_01 @ coefs_cvd - center_cvd))

    fem = sex == 2
    mal = sex == 1
    imp_f = int(np.sum(fem & (risk_cvd_bug > risk_all)))
    imp_f_fix = int(np.sum(fem & (risk_cvd_fix > risk_all)))
    imp_m = int(np.sum(mal & (risk_cvd_bug > risk_all)))

    M_bug = np.column_stack([np.ones(n), ckmai_pub, age, sex, col("Smoking"), col("Diabetes"), col("Hypertension"), col("EGFR"), col("UACR")])
    M_fix = np.column_stack([np.ones(n), ckmai_pub, age, (sex == 1).astype(float), col("Smoking"), col("Diabetes"), col("Hypertension"), col("EGFR"), col("UACR")])
    p_ckm_bug = 1.0 / (1.0 + np.exp(-(M_bug @ glm_coefs)))
    p_ckm_fix = 1.0 / (1.0 + np.exp(-(M_fix @ glm_coefs)))

    # Write feature sign-inversion CSV
    with open(out_dir / "ckmai_feature_sign_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "rds_published_weight", "true_cox_beta_z", "true_cox_se_z", "z_stat", "sign_inverted"])
        for j, feat in enumerate(features):
            w.writerow([feat, f"{weights_abs[j]:.6f}", f"{beta_z[j]:.6f}", f"{se_z[j]:.6f}", f"{beta_z[j]/se_z[j]:.3f}", bool(beta_z[j] < 0)])

    summary = {
        "doi": "10.1371/journal.pmed.1005078",
        "n_nhanes": n,
        "n_female": int(fem.sum()),
        "n_male": int(mal.sum()),
        "official_hashes": hashes,
        "code_r_line_648": line_648,
        "sign_inversion": {
            "n_flipped_protective_features": int(np.sum(beta_z < 0)),
            "flipped_features": [features[j] for j in range(len(features)) if beta_z[j] < 0],
            "all_cause_c_published": round(harrell_c(time_v, mort, ckmai_pub), 4),
            "all_cause_c_signed": round(harrell_c(time_v, mort, ckmai_signed), 4),
            "all_cause_c_5fold_oof_signed": round(harrell_c(time_v, mort, oof_sgn), 4),
            "cvd_c_published": round(harrell_c(time_v, cvd, ckmai_pub), 4),
            "cvd_c_age_alone": round(harrell_c(time_v, cvd, age), 4),
            "cvd_c_signed": round(harrell_c(time_v, cvd, ckmai_signed), 4),
            "cvd_c_5fold_oof_signed": round(harrell_c(time_v, cvd, oof_sgn), 4),
            "age_sex_adjusted_lrt_published": round(lrt_pub, 2),
            "age_sex_adjusted_lrt_signed": round(lrt_sgn, 2),
            "pct_lrt_destroyed": round(100.0 * (1.0 - lrt_pub / lrt_sgn), 2),
            "pearson_r_ckmai_age": round(float(np.corrcoef(ckmai_pub, age)[0, 1]), 4),
        },
        "sex_encoding_bug": {
            "nomogram_allcause_sex_coef": round(float(coefs_all[cnames_all.index("Sex")]), 6),
            "nomogram_cvd_sex_coef": round(float(coefs_cvd[cnames_cvd.index("Sex")]), 6),
            "logistic_ckm_high_sex_coef": round(float(glm_coefs[3]), 6),
            "female_impossible_cvd_gt_allcause_n": imp_f,
            "female_impossible_cvd_gt_allcause_pct": round(100.0 * imp_f / fem.sum(), 2),
            "female_impossible_after_fix_n": imp_f_fix,
            "male_impossible_n": imp_m,
            "female_mean_10y_cvd_true_pct": round(100.0 * float(risk_cvd_fix[fem].mean()), 2),
            "female_mean_10y_cvd_buggy_pct": round(100.0 * float(risk_cvd_bug[fem].mean()), 2),
            "female_10y_cvd_inflation_ratio": round(float(risk_cvd_bug[fem].mean() / risk_cvd_fix[fem].mean()), 2),
            "female_10y_cvd_median_ratio": round(float(np.median(risk_cvd_bug[fem] / risk_cvd_fix[fem])), 2),
            "female_mean_ckm_high_true_pct": round(100.0 * float(p_ckm_fix[fem].mean()), 2),
            "female_mean_ckm_high_buggy_pct": round(100.0 * float(p_ckm_bug[fem].mean()), 2),
            "female_ckm_high_inflation_ratio": round(float(p_ckm_bug[fem].mean() / p_ckm_fix[fem].mean()), 2),
            "overall_ckm_high_auc_true": round(roc_auc(ckm_high, p_ckm_fix), 4),
            "overall_ckm_high_auc_buggy": round(roc_auc(ckm_high, p_ckm_bug), 4),
        },
    }
    (out_dir / "ckmaicalc_safety_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
