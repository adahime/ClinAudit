# Independent Adversarial Forensic Audit: Chen et al. (2026)

- **Paper:** Chen et al., *"Multi-omics fusion with machine learning enables robust prediction of treatment response in ovarian cancer for precision population health"*, *npj Digital Medicine* (2026-07-14)
- **Identifiers:** DOI `10.1038/s41746-026-02991-x` | PMCID `PMC13737307` | PMID `42443519`
- **Verification Script:** `independent_verify_02991x.py` (Python 3.15.0 stdlib + NumPy 2.5.4 only; zero external ML/spreadsheet libraries)
- **Structured Output:** `independent_verification_report.json` | **File Hashes:** `OFFICIAL_HASHES.json`
- **Final Verdict:** **`CONFIRMED`** — Catastrophic, undisclosed train–test exact-clone data leakage (63.2%–68.7% of all test rows across all four modalities), post-oversampling cross-validation leakage in Supplementary Fig. 5, and collapse of true out-of-sample discrimination from published AUC `0.928–0.962` to `~0.49–0.72` (with severe non-responder specificity collapse) are independently confirmed from scratch.

---

## 1. Official File Provenance (`OFFICIAL_HASHES.json`)

| File (in `official_files/`) | Bytes | SHA-256 | MD5 |
|---|---:|---|---|
| `41746_2026_2991_MOESM4_ESM.xlsx` | 6,995,030 | `16a82a0a69dee18e1ccd2ca473d903118b86f44ae45041817dd76e478b03f8f9` | `5e50aff845ddb4a3f36229a53cbd3732` |
| `41746_2026_2991_MOESM1_ESM.pdf` | 5,296,983 | `8794c5678b577bd180ecb73f2b340b5db31e4c7d06abc83d95aeb3c03e3e9f7b` | `9eddaf140b537d8d026e0022bfd5769e` |
| `epmc_PMC13737307_fulltext.xml` | 153,343 | `d093413d100da289494cf08d8dcbce70f1e42c43daae3e89cd841a12e57a21d7` | `a8145b9f2cfa6f0722c9ec8aeb160b9e` |
| `springer_article.html` | 469,333 | `f3cfde0664aafbbd2489b6a86aaca61e7235f2116bdbc8d4ac75220b6ad4b2a8` | `0f5b80d61fa5ec362f0678be113b3b4a` |
| `TCGA-OV-Drug-Response-Prediction_main.zip` | 576,412 | `dfae76daf9451c09e58a74bd676d95bf3f3722eafed3ebc465a54373886e8af2` | `4677d85bb86e9447f3246b23bad2215d` |

Note: The official GitHub repository contains only inference scripts (`run_*.py`), serialized `.pkl` models/scalers, and `best_features_*.csv` lists — no training or splitting code is published.

---

## 2. Verbatim Disclosure Audit (`epmc_PMC13737307_fulltext.xml` & `MOESM1_ESM.pdf`)

Full-text search of the published article and `MOESM1_ESM.pdf` yields **0 occurrences** of `duplicat*`, `overlap`, `patient-level`, `GroupKFold`/`GroupShuffleSplit`, `aliquot`, or `multiple regimen/drug/sample`. The single occurrence of `leak*` is an explicit claim that leakage was prevented. Every key methodological statement is directly contradicted by `MOESM4_ESM.xlsx` and `MOESM1_ESM.pdf`:

1. **Methods — Study cohort and data acquisition:**
   > *"After matching transcriptomic, RPPA proteomic, DNA methylation, treatment-response, and pharmaceutical-treatment information across data layers, 375 patients were retained as the matched multi-omics modeling cohort for model development and evaluation."*
   - **Reality:** `375 = 279 + 96` double-counts 62 rows copied verbatim into both train and test. Removing cross-split duplicates yields 313 rows, **224** unique 143-feature vectors, and **210** patient-level connected components (after grouping multi-aliquot Cartesian blocks such as 2×2 methylation × transcriptome pairs).

2. **Methods — Data splitting:**
   > *"To enable unbiased model evaluation, the matched multi-omics modeling cohort (n = 375) was randomly partitioned into a training set (n = 279) and an independent held-out test set (n = 96) using a fixed random seed of 42, corresponding to an approximate 74.4%/25.6% train-test split."*
   - **Reality:** **62 / 96 (64.58%)** of Early Fusion test rows are bit-for-bit identical (both raw XML string and `1e-12` float) to pre-ADASYN training rows, and appear in the **exact same relative row order**.

3. **Methods — OMICS-FUSE equations:**
   > *"The independent held-out test set contained 96 real patients, retained a 96 × 143 feature matrix, and was not oversampled."*
   - **Reality:** The 96 test rows contain only **56** unique feature vectors (including one 12-copy clone and four 3-copy clones) and only **32** unique unseen feature vectors (**34** unseen rows).

4. **Methods — Model training and hyperparameter optimization:**
   > *"To prevent information leakage, all imputation, scaling, feature selection, and oversampling steps were fitted exclusively within the training data. During cross-validation, oversampling was applied only to the training fold and never to validation folds or the held-out test samples."*
   - **Reality:** Supplementary Fig. 5a–d learning-curve X-axes reach `~328` (`0.8 × 411`), `~427` (`0.8 × 534`), `~443` (`0.8 × 554`), and `~675` (`0.8 × 844`), proving 5-fold CV was run directly on the **post-ADASYN / post-SMOTE** oversampled matrices (not `0.8 × 279 = 223`, `0.8 × 364 = 291`, `0.8 × 409 = 327`, `0.8 × 603 = 482`). In Supplementary Fig. 5b (Transcriptome), validation AUC stays at `0.66–0.75` for `n ≤ 256` (all pre-SMOTE rows) and jumps vertically to `0.91 → 0.97` immediately after `n` crosses `291 = 0.8 × 364` into the appended SMOTE tail.

5. **Results — Learning-curve analysis:**
   > *"As the size of the training set increased, the training AUC gradually decreased while the validation AUC steadily improved, with both curves converging toward similar values."*
   - **Reality:** In Supplementary Fig. 5a–d, the blue **Training ROC-AUC** curve is **flat at `1.000`** across every training size in all four modalities.

---

## 3. Exact Feature-Vector Overlap, Contradictions & Multiplicity Parity Proof

Every sheet in `MOESM4_ESM.xlsx` was parsed directly from XML. In all four training sheets, ADASYN/SMOTE synthetic `y = 0` rows occupy a contiguous trailing all-`0` block after the last `y = 1` row, allowing exact separation of pre-oversampling real rows (`train_pre`) from synthetic rows (`train_post`). All values are already `RobustScaler`-transformed on `train_full` (`max|median| = 0.000000` across all 4 modalities).

| Modality | Feats | Train (`pre` / `full`) | Test (`y=0` / `y=1`) | Exact Test Overlap (raw = `1e-12`) | Unseen Test Rows (`uniq`) | Unique Real Pool (`excl. contra`) | Contradictory-Label Vectors | Min Unseen Distance (`L_inf` / `L2`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Early Fusion** | 143 | 279 / 411 | 96 (18 / 78) | **62 / 96 (64.58%)** | 34 (32: 12 neg, 20 pos) | 224 (210 aliquot groups) | 0 | 2.51 / 9.37 |
| **Methylation** | 256 | 603 / 844 | 249 (79 / 170) | **171 / 249 (68.67%)** | 78 (75: 24 neg, 51 pos) | 474 (475 incl. 1 contra) | **1** (`tr_y0:1, tr_y1:1, te_y0:1, te_y1:1`) | 9.51 / 23.48 |
| **Proteomics** | 240* | 409 / 554 | 183 (68 / 115) | **122 / 183 (66.67%)** | 61 (58: 18 neg, 40 pos) | 325 | 0 | 1.53 / 6.63 |
| **Transcriptome** | 191 | 364 / 534 | 114 (20 / 94) | **72 / 114 (63.16%)** | 42 (40: 14 neg, 26 pos) | 303 | 0 | 2.67 / 10.78 |

\*Proteomics XLSX contains 240 features vs. 245 in Methods and GitHub (`best_features_XGBoost_pharma_yes_proteomics.csv`); see Caveat C2.

### Multiplicity Parity (`k_train_pre == k_test`) & Exact Root-Cause Mechanism

Across **all four modalities**, **142 / 142 (100.0%)** of shared feature vectors exhibit **exact pre-oversampling multiplicity parity `k_train_pre == k_test`** — including `(12, 12)`, `(15, 15)`, `(6, 6)`, and `(24, 24)` — with **0 / 216** total-multiplicity-2 vectors splitting `(1, 1)` and **100% identical row ordering** of shared rows between train and test (`tr_shared_seq == te_shared_seq == True` in all 4 modalities):

| Modality | Shared Vectors with `k_tr_pre == k_te` | Max Shared Multiplicity (`pre` → `post`) | `M=2` Vectors Split `(1,1)` | `P(0 of M=2 split 1-1 \| random)` | `P(all shared 50/50 \| shared, random)` | Observed `(2,0):(2,2):(0,2)` vs. 80/20 `.isin()` Bug (`χ²`, df=2) |
|---|---:|---|---:|---:|---:|---|
| **Early Fusion** | **24 / 24 (100%)** | `(12, 12)` | **0 / 39** | `10^-8.12` | **`10^-14.84`** | `[37, 19, 2]` vs. `[37.12, 18.56, 2.32]` (`χ² = 0.055`, p=0.97) |
| **Methylation** | **56 / 56 (100%)** | `(24, 24)` → `(54, 24)` | **0 / 78** | `10^-18.09` | **`10^-34.13`** | `[75, 43, 3]` vs. `[77.44, 38.72, 4.84]` (`χ² = 1.249`, p=0.54) |
| **Proteomics** | **34 / 34 (100%)** | `(24, 24)` → `(49, 24)` | **0 / 48** | `10^-11.61` | **`10^-21.55`** | `[45, 25, 3]` vs. `[46.72, 23.36, 2.92]` (`χ² = 0.181`, p=0.91) |
| **Transcriptome** | **28 / 28 (100%)** | `(12, 12)` | **0 / 51** | `10^-10.00` | **`10^-18.35`** | `[49, 22, 2]` vs. `[46.72, 23.36, 2.92]` (`χ² = 0.480`, p=0.79) |

**Exact Root Cause Proven:** Splitting a **non-deduplicated** `submitter_id` sequence (containing repeated clinical/pharma rows and multi-aliquot Cartesian merges) with an 80/20 `train_test_split`, then subsetting via `df[df['submitter_id'].isin(train_ids)]` and `df[df['submitter_id'].isin(test_ids)]`. Whenever a patient with `m ≥ 2` rows has at least one row in `train_ids` and one in `test_ids`, `.isin()` pulls **all `m` rows** into both splits in identical order — deterministically forcing `(m, m)` parity, exact sequence equality, zero `(1, 1)` splits, `[0.64 : 0.32 : 0.04]` ratios for `m=2`, and explaining why an 80/20 split produced an apparent "74.4%/25.6%" (`279/375`) split! Additionally, ADASYN/SMOTE then interpolated between identical minority-class duplicates, generating `11 / 63 / 44 / 15` exact-clone synthetic training rows (`2 / 43 / 32 / 1` exact clones of test rows), inflating `(6,6)→(17,6)` and `(24,24)→(54,24)` in Methylation and `(6,6)→(12,6)` and `(24,24)→(49,24)` in Proteomics.

---

## 4. Independent Pure-NumPy Performance Evaluation

Using clean from-scratch pure-NumPy implementations of Random Forest (`200` CART Gini trees), exact IRLS/Newton L2 Logistic Regression (`C=1.0`, `C=0.1`), and Euclidean 5-NN (with 2,000-resample bootstrap 95% CIs):

| Modality | Published Table 1 AUC | Regime (a): Full Leaky Test AUC (RF / LR `C=0.1`) | Regime (b): Leaked Clones Only AUC (RF / LR) | Regime (c): Truly Unseen Unique Test AUC `[95% CI]` & Spec (RF) | Regime (c): Truly Unseen Unique Test AUC `[95% CI]` & Spec (LR `C=0.1`) | Regime (d): Strict 5-Fold Dedup CV OOF AUC `[95% CI]` (RF_unw / RF_bal / LR_bal `C=0.1`) |
|---|---:|---|---|---|---|---|
| **Early Fusion** | **0.939** (RF) | **0.9491** / 0.7799 | **1.0000** / **1.0000** | **0.7188** `[0.5020, 0.9179]`, Spec **0.3333** (4/12) | **0.5250** `[0.3000, 0.7580]`, Spec 0.4167 | **0.6551** `[0.5795, 0.7281]` / **0.6328** / **0.6482** *(210-group CV: 0.6468 / 0.6303 / 0.6376)* |
| **Methylation** | **0.955** (CatBoost) | **0.9477** / 0.8940 | **0.9999** / **0.9999** | **0.4910** `[0.3533, 0.6296]`, Spec **0.0417** (1/24) | **0.5180** `[0.3765, 0.6542]`, Spec 0.3333 | **0.6228** `[0.5689, 0.6756]` / **0.5538** / **0.7866*** *(see C1 below!)* |
| **Proteomics** | **0.962** (XGBoost) | **0.9533** / 0.9247 | **1.0000** / **1.0000** | **0.5208** `[0.3236, 0.7140]`, Spec **0.2222** (4/18) | **0.6472** `[0.4789, 0.7954]`, Spec 0.4444 | **0.5504** `[0.4819, 0.6184]` / **0.5955** / **0.6213** |
| **Transcriptome** | **0.928** (RF) | **0.9223** / 0.7553 | **1.0000** / **1.0000** | **0.6168** `[0.4160, 0.8217]`, Spec **0.0714** (1/14) | **0.5769** `[0.3854, 0.7635]`, Spec 0.2857 | **0.5818** `[0.5116, 0.6511]` / **0.6077** / **0.5961** |

\* **Crucial Proof of Pre-CV LASSO Feature-Selection Bias (Caveat C1):** In Methylation Regime (d), `LogReg_L2_C0.1_balanced` gets OOF AUC `0.7866` overall ONLY because those 256 CpGs were pre-selected out of 20,280 CpGs by LASSO on `train_pre`: splitting the exact same OOF predictions gives AUC **`0.8345`** on the `399` patients exposed to LASSO vs. **`0.5400`** on the `75` patients never seen by LASSO (matching Regime c's `0.5180`!). Likewise in Early Fusion, `LogReg` OOF AUC is `0.6861` on the `192` LASSO-exposed patients vs. `0.4750` on the `32` never-exposed patients. In Proteomics (unsupervised variance filtering, no LASSO), there is zero gap (`0.6185` vs. `0.6361`). Thus **Regime (c)** is the only 100% leak-free evaluation of the published feature matrices.

---

## 5. `MOESM1_ESM.pdf` Confusion Matrices & Non-Responder Specificity Bounds

High-resolution extraction of `MOESM1_ESM.pdf` Supplementary Figs. 1–4 verified the exact published confusion matrices for all Table 1 flagship models. Crucially, **Table 1 omitted Specificity completely** (reporting only Accuracy, Precision, Recall, F1, AUC). Let `T = TN_total = TN_leaked + TN_unseen`, `L0` = leaked test `y=0` rows, `U0` = unseen test `y=0` rows:

| Modality (Table 1 Model) | Published CM `[[TN, FP], [FN, TP]]` | Overall Specificity `T/(L0+U0)` | `L0` (`unambig`) / `U0` | Unconditional Bounds `[max(0,T-L0)/U0, min(T,U0)/U0]` | Monotonicity Upper Bound (`Spec_leaked ≥ Spec_unseen`) | Memorization-Implied Unseen Specificity (`TN_leaked = L0_unambig`, Train AUC=1.0) |
|---|---|---:|---:|---:|---:|---:|
| **Early Fusion (RF)** | `[[9, 9], [1, 77]]` | **50.0%** (9/18) | 6 (6) / 12 | `[25.0%, 75.0%]` | **≤ 50.0%** | **25.0% (3/12)** |
| **Transcriptome (RF)** | `[[9, 11], [3, 91]]` | **45.0%** (9/20) | 6 (6) / 14 | `[21.4%, 64.3%]` | **≤ 45.0%** | **21.4% (3/14)** |
| **Proteomics (XGBoost)** | `[[54, 14], [2, 113]]` | **79.4%** (54/68) | 50 (50) / 18 | `[22.2%, 100.0%]` | **≤ 79.4%** | **22.2% (4/18)** |
| **Methylation (CatBoost)** | `[[57, 22], [6, 164]]` | **72.2%** (57/79) | 55 (54) / 24 | `[8.3%, 100.0%]` | **≤ 72.2%** | **12.5% (3/24)** *(or 8.3% = 2/24 if contra predicted 0)* |

**Additional Internal Reporting Discrepancies Identified in `MOESM1_ESM.pdf` vs. Main Text:**
- **Table 2 TN/FP Transposition:** All five Top-20 confusion matrices in main-text Table 2 swap TN and FP relative to Supplementary Fig. 1b and Table 2's own Accuracy/F1 columns (e.g., `RandomForest_RF_Top20` prints `[[7,11],[3,75]]` [implied Acc `0.854`] vs. Supp Fig. 1b `[[11,7],[3,75]]` [Acc `0.896`, matching Table 2]).
- **Table 1 Proteomics XGBoost Precision/Recall Mismatch:** Table 1 prints Precision `0.881` and Recall `0.965` (which yield F1 `0.921`, nearly identical to Methylation CatBoost's `0.882 / 0.965`), contradicting Table 1's own F1 `0.934` and Supplementary Fig. 3's `[[54,14],[2,113]]` (actual Precision `0.890`, Recall `0.983`, F1 `0.934`).

---

## 6. Live Public Record Status

- **Crossref API (`10.1038/s41746-026-02991-x`):** `status: ok`, published `2026-07-14`, `update-to: None`, `updated-by: None`, `relation: {}`, `is-referenced-by-count: 2`.
- **Europe PMC API (`PMC13737307`):** `commentCorrectionList: None`, `pubType: ['research-article', 'Journal Article']`.
- **PubPeer:** Direct programmatic API returned Cloudflare `403 Forbidden`; live web index search on `pubpeer.com` and general web search confirmed **0 prior PubPeer comments, corrections, expressions of concern, or retractions**.

---

## 7. Adversarial Caveats & Refuted Benign Explanations

- **C1 (Pre-CV LASSO Feature Selection in Regime d):** Properly accounted for via LASSO-exposure OOF split; Regime (c) (unseen unique test patients) has zero feature-selection and zero patient leakage.
- **C2 (Proteomics 240 vs. 245 features):** `MOESM4_ESM.xlsx` has 240 proteomics features (`Gli3` present; `ERRalpha, Lasu1, RPA32, SHP2_pY542, YAP, p38-a` absent compared to GitHub's 245). Exact leakage and parity findings are unaffected.
- **C3 (Table 1 & Table 2 Internal Discrepancies):** Documented independently above; secondary to the primary data leakage.
- **C4 (Mathematical Distinction of Specificity Bounds):** Explicitly distinguishes unconditional bounds `[max(0, T-L0)/U0, min(T, U0)/U0]`, monotonicity upper bounds `Spec_unseen ≤ Spec_total`, and memorization-implied unseen specificity `(T - L0_unambig)/U0` (directly corroborated by Supp Fig. 5 Training AUC = `1.000` and our empirical RF unseen specificity of `4/12`, `1/24`, `4/18`, `1/14`).
- **C5 (Benign "Multiple Regimens per Patient" Hypothesis Refuted):** Fails on four independent grounds: (i) zero disclosure (explicitly described as `375` / `279` / `96` real patients); (ii) splitting identical pre-treatment omics profiles and patient-level response labels across train and test is still severe patient-level leakage; (iii) multi-aliquot 2×2 Cartesian blocks occur; and (iv) 142/142 exact `(k, k)` parity + 0/216 `(1,1)` splits + 100% identical row order is impossible under random row splitting (`p ≤ 10^-14.84` per modality) and matches the 80/20 `.isin()` non-unique ID split bug (`χ² = 0.055–1.249`, df=2, all `p > 0.53`).
- **C6 (Scope Boundary):** This audit targets the computational dataset, ML evaluation, and supplementary figures; it makes no independent claim about the downstream cell-line RT-qPCR/ELISA bench assays (though those 6 hub genes were selected downstream of SHAP on the leaked models).
