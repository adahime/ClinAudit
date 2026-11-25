# Supplementary Appendix

**Manuscript:** Automated Pre-Publication Forensic Auditing of Clinical Machine Learning Studies: A 2026 Cross-Journal Empirical Evaluation and Open-Source Verification Suite
**Author:** Xingyu Zhang (Independent Researcher; ORCID: 0009-0005-6181-7465)

---

## Supplementary Table S1. Exact Accounting of the 10 Records Excluded After HTML Abstract Recovery (316 → 306 Eligible 2026 Studies)

| # | DOI | Journal | Exclusion Criterion | Title |
|---:|---|---|---|---|
| 1 | `10.1038/s41467-026-79690-4` | nature_communications | `no_quantitative_prediction_metric` | A multimodal ultrasound foundation model for diagnosis of unexplained lymphadenopathy |
| 2 | `10.1038/s41746-026-03047-w` | npj_digital_medicine | `no_quantitative_prediction_metric` | Shaping the future of cancer screening with artificial intelligence-empowered multi-cancer early detection |
| 3 | `10.1038/s41746-026-03132-0` | npj_digital_medicine | `no_quantitative_prediction_metric` | Clinicians vs. artificial intelligence in patient outcome prediction in the intensive care unit |
| 4 | `10.1038/s41746-026-03151-x` | npj_digital_medicine | `no_quantitative_prediction_metric` | Full end-to-end diagnostic workflow automation of 3D OCT via foundation model-driven AI for retinal diseases |
| 5 | `10.1038/s41746-026-03154-8` | npj_digital_medicine | `no_quantitative_prediction_metric` | Modernizing evidence generation for clinical artificial intelligence: the case for adaptive platform trials |
| 6 | `10.1038/s41746-026-03207-y` | npj_digital_medicine | `no_quantitative_prediction_metric` | Automated pancreatic cancer pathology image segmentation using deep learning to quantify lymphocyte stroma ratio |
| 7 | `10.1038/s41746-026-03156-6` | npj_digital_medicine | `no_quantitative_prediction_metric` | Vision wearables with artificial intelligence to close the sensory gap in patient characterization |
| 8 | `10.1038/s41746-026-03426-3` | npj_digital_medicine | `no_quantitative_prediction_metric` | RETBind: a patient-level multimodal foundation model for unified cross-modal retinal representation learning |
| 9 | `10.1038/s41746-026-03463-y` | npj_digital_medicine | `no_quantitative_prediction_metric` | Deep learning retinal vascular biomarker predicts preeclampsia in a multicenter prospective cohort |
| 10 | `10.1038/s41746-026-03467-8` | npj_digital_medicine | `no_quantitative_prediction_metric` | Multimodal CT-pathology imaging deep learning model for risk stratification in resected stage I lung adenocarcinoma |

---

## Supplementary Table S2. Complete Line-by-Line Adjudication of All 20 Studies Flagged by Uncalibrated `ClinAudit v0.1.0` Across the 174 Downloaded 2026 Papers

| # | DOI | Journal | Triggered Rules | Adjudication Verdict | Technical & Disclosure Basis |
|---:|---|---|---|---|---|
| 1 | `10.1038/s41467-025-67436-7` | nature_communications | `STAT-004` | `FALSE_POSITIVE` | Metabolomics feature-association regression table (beta/OR/pval), not ML model-comparison P-values. |
| 2 | `10.1038/s41467-025-67712-6` | nature_communications | `STAT-004` | `FALSE_POSITIVE` | Proteomics feature-association Wilcoxon table (NPX proteins vs sex, N=1142), not bootstrap-pseudoreplication model comparison. |
| 3 | `10.1038/s41467-026-68355-x` | nature_communications | `STAT-003` | `FALSE_POSITIVE` | Different subgroup/adjustment Cox HR tables matched by generic label 'dm'. |
| 4 | `10.1038/s41467-026-68415-2` | nature_communications | `STAT-002/003/004/005` | `FALSE_POSITIVE` | Proteomics cohort summary & association tables (N=4024), not ML discrimination comparisons. |
| 5 | `10.1038/s41467-026-68956-6` | nature_communications | `EVAL-002/STAT-002/004` | `FALSE_POSITIVE` | KM t=0 survival_prob_pct=100.0 [100.0,100.0] boundary CI; Cox HR per-SD P=0 underflow on UK Biobank; non-DCA table misidentified as DCA. |
| 6 | `10.1038/s41467-026-70077-z` | nature_communications | `STAT-001/002/003` | `CONFIRMED_MINOR_REPORTING_DEFECT` | Supplementary Data 7 row 23 (JSIEC FM-MEH-SDPP) prints inverted 95% CI '0.996 (0.995, 0.887)' (upper bound 0.887 < lower bound 0.995), plus zero-width '0.988 (0.988, 0.988)' in Supp Data 3 & 6. |
| 7 | `10.1038/s41467-026-70584-z` | nature_communications | `STAT-002` | `CONFIRMED_UNDISCLOSED_METHODOLOGICAL_DEFECT` | Prospective blind discovery: aipal_validation/ml/test.py:206-207 computes ci_width = norm.ppf(0.975) * (std_dev / np.sqrt(2000)), shrinking 95% CIs by 44.72x across adult & pediatric international leukemia cohorts (166+ zero-width CIs in Suppl Data 1 & 4 and main text '0.82 [0.82-0.82]', '0.95 [0.95-0.95]'). |
| 8 | `10.1038/s41467-026-73023-1` | nature_communications | `DEPLOY-004` | `FALSE_POSITIVE` | Regex without word boundary matched 'abs(' inside ggplot2 'labs(x=..., y=...)'. Addressed in v0.2.0. |
| 9 | `10.1038/s41467-026-73878-4` | nature_communications | `STAT-001` | `FALSE_POSITIVE` | Columns 'yerr_lower' and 'yerr_upper' are matplotlib asymmetric error-bar half-heights, not [lo, hi] CI bounds. Addressed in v0.2.0. |
| 10 | `10.1038/s41467-026-74209-3` | nature_communications | `EVAL-006` | `VALIDATION_NAMING_AMBIGUITY_TRIAGE_CONTROL` | Triage control (reclassified upon full-text Methods review): four training scripts select checkpoints by minimizing loss_t on variables named x_test/min_test_loss (test_size=0.1/0.2), but Methods confirm that for 3/4 scripts (CLIP_pre_train.py, CLIP_fine_tune.py, T2Pdecoder_VAE_train.py) this split served as an internal validation split paired with independent external 'Unseen' (n=5,608) and TSKGA test cohorts (only T2Pdecoder_VAE_ft_cite.py evaluates vae.decode(x_test) on the same 20% split). Illustrates why static EVAL-006 flags require Methods verification. |
| 11 | `10.1038/s41467-026-76071-9` | nature_communications | `EVAL-003` | `FALSE_POSITIVE` | Vendored AutoGluon utility (ag_metrics.py) where log_loss sets needs_proba=True, bypassing the else argmax branch. |
| 12 | `10.1038/s41467-026-78075-x` | nature_communications | `EVAL-006` | `FALSE_POSITIVE` | UpTCR selects best_val strictly on val_pearson_corr/val_aupr; 'if test_loader is not None: self.test(test_loader, best_val)' only evaluates/logs test metrics. Addressed in v0.2.0. |
| 13 | `10.1038/s41746-025-02305-7` | npj_digital_medicine | `STAT-004/002` | `FALSE_POSITIVE` | Training-set DeLong Z=-8.11 (AUC 0.8965 vs 0.6067, large delta=0.29) yields genuine P=4.98e-16; boundary zero-width p_value=1.0. |
| 14 | `10.1038/s41746-026-02527-3` | npj_digital_medicine | `DEPLOY-004/STAT-004/005` | `FALSE_POSITIVE` | MaAsLin2 volcano-plot filter 'filter(abs(coef) > 0.002)' is differential-abundance filtering, not risk-score weighting; Supp Tables 2-3 are microbiome feature p-values. Addressed in v0.2.0. |
| 15 | `10.1038/s41746-026-02707-1` | npj_digital_medicine | `STAT-004` | `FALSE_POSITIVE` | Ordinal logistic regression predictor coefficient table (large-N physiological features, z=-25.2 -> P=5.87e-140), not ML model comparison. |
| 16 | `10.1038/s41746-026-02964-0` | npj_digital_medicine | `EVAL-004` | `CONFIRMED_TECHNICAL_MATCH_EXPLICITLY_DISCLOSED` | age_estimation.ipynb calls StandardScaler().fit_transform(test_data), but full-text Discussion explicitly discloses: 'A notable limitation of the current study is independently scaling external datasets for age estimation, which was necessitated by changes in hardware and firmware.' Demonstrates necessity of full-text disclosure triage. |
| 17 | `10.1038/s41746-026-02968-w` | npj_digital_medicine | `EVAL-005` | `FALSE_POSITIVE` | Boolean flag 'if have_future:' checks existence of prospective incident-dementia cohort file rather than post-baseline treatment stratification. Addressed in v0.2.0. |
| 18 | `10.1038/s41746-026-02991-x` | npj_digital_medicine | `LEAK-001/LEAK-003` | `CONFIRMED_UNDISCLOSED_METHODOLOGICAL_DEFECT` | Blind detection of reference Case 1: 63.2%-68.7% exact 17-digit test clones across all 4 omics modalities, 142/142 exact (k,k) mirror parity, 1 contradictory-label profile, unseen AUC collapse from 0.93-0.96 to 0.49-0.72. |
| 19 | `10.1371/journal.pmed.1004770` | plos_medicine | `EVAL-003` | `FALSE_POSITIVE` | R helper trimvals() uses ifelse(x >= trims[1] & x <= trims[2], x, NA) to Winsorize/trim continuous outliers, not {0,1} binarization. Addressed in v0.2.0. |
| 20 | `10.1371/journal.pmed.1005190` | plos_medicine | `EVAL-003` | `CONFIRMED_UNDISCLOSED_METHODOLOGICAL_DEFECT` | Prospective blind discovery: across all 5 XGBoost evaluation scripts (e.g., MultivariableXGBoost_testing_best_unique_35f.py), main() directly invokes (1) calculate_calibration() (L37-57, L173), binarizing predicted survival times into hard {0,1} y_pred_binary passed to linregress() and 10-bin CalibrationDisplay.from_predictions(); (2) calculate_concordance_index() (L65-85, L176), dropping all event-free survivors beyond horizon (mask = y_true < days) and calling concordance_index(y_true_period, y_pred_period) without event_observed; and (3) overall concordance_index(y_true, y_pred) at L158 also omitting event_observed (despite passing event_observed=(y_upper == y_lower) to kmf.fit() at L165). Although the published article noted that the AFT XGBoost comparator did not allow individual probability calculations and could therefore not be calibrated in primary plots, all five official evaluation scripts still executed the binarized calibration routine directly inside main() alongside the uncensored and survivor-dropping C-index calculations. |

---

## Supplementary Table S3. Complete Out-of-Time Late-2025 External Validation Adjudication (Frozen `ClinAudit v0.2.0`)

Window: 2025-09-01 to 2025-12-31 | Screened N0 = 759 | Eligible N1 = 95 | Downloaded N = 46 (522 files), directly parsed N = 45 (514 supported files; 1 zip-only verified 0 alerts on extraction; 0 crashes; median 368.0 ms/paper; flagged 4/45 = 8.89% parsed / 4/46 = 8.7% downloaded).

| # | DOI | Journal | Rules | Adjudication Verdict | Findings |
|---:|---|---|---|---|---|
| 1 | `10.1038/s41467-025-63846-9` | nature_communications | `STAT-001` | `FALSE_POSITIVE` | esm_1.xlsx sheet 'figure4' vertically stacks multiple distinct sub-tables ((a) Cox model rows 0-48, (c) LogRank tests rows 49-55, (d) TCGA LogRank rows 56-60, (e) SHAP decision values rows 61+) inside the same Excel columns, causing STAT-001 to read LogRank/SHAP values beneath the Cox CI headers. Illustrates the need for one-table-per-sheet Source Data formatting. |
| 2 | `10.1038/s41746-025-02189-7` | npj_digital_medicine | `STAT-004` | `FALSE_POSITIVE` | esm_2.xlsx Sheet1 is a GSVA biological-process correlation table (Features vs Related Process, Pearson r, p_value), not an ML model-comparison table. |
| 3 | `10.1038/s41467-025-66483-4` | nature_communications | `STAT-004` | `FALSE_POSITIVE` | esm_3.xlsx Figure 2b is a Gene Ontology (GOTERM_BP_DIRECT) enrichment table (PValue, FDR), not an ML model-comparison table. |
| 4 | `10.1038/s41467-025-67656-x` | nature_communications | `EVAL-004, STAT-002, STAT-004` | `CONFIRMED_TECHNICAL_MATCH_AND_COLLATERAL_UNDISCLOSED_BUG` | EVAL-004 accurately flagged RandomUnderSampler().fit_resample(x_test/x_test_male/x_test_female) in UKB_by_sex_and_undersampling.py (L150, 324, 377). Full-text XML (PMC12830939) confirmed balanced test undersampling was disclosed as a sensitivity analysis—yet inspecting that flagged block exposed an undisclosed constant-seed bootstrap bug across all 11 loops in UKB_by_sex_and_undersampling.py and external_validation.py (resample(..., random_state=1996)), collapsing np.std() to IEEE-754 float64 machine epsilon (1.11-4.44e-16; 21 times) and subnormal underflow (2.06e-296; 4 times)—25 zero-variance uncertainty estimates printed verbatim in the published main text (0 corrections on Crossref). |

---

## Supplementary Table S4. Fifty-Study Random Unflagged Negative-Control Audit Log

Sampling seeds: `2026` (initial n = 30) and `2026050` (extension n = 20) from 154 unflagged studies. Result: 0/50 additional defects within accessible artifacts under pre-specified rules (NPV = 100.0%; false omission rate = 0.0%; exact two-sided 95% Clopper–Pearson upper bound = 7.11%; coverage: 39/50 with code, 25/50 with tables, 2/50 zip-only archives verified clean on secondary extraction).

| # | DOI | Journal | Files | Audit Result & Notes |
|---:|---|---|---:|---|
| 1 | `10.1038/s41467-025-66758-w` | nature_communications | 1 | Clean (0 missed defects) |
| 2 | `10.1038/s41467-025-68198-y` | nature_communications | 7 | Clean (verified SMOTE/RandomOverSampler strictly inside training split/pipeline) |
| 3 | `10.1038/s41467-026-72713-0` | nature_communications | 5 | Clean (0 missed defects) |
| 4 | `10.1038/s41467-026-73021-3` | nature_communications | 10 | Clean (verified SMOTE/RandomOverSampler strictly inside training split/pipeline) |
| 5 | `10.1038/s41467-026-73088-y` | nature_communications | 41 | Clean (0 missed defects) |
| 6 | `10.1038/s41467-026-73636-6` | nature_communications | 10 | Clean (0 missed defects) |
| 7 | `10.1038/s41467-026-74077-x` | nature_communications | 40 | Clean (0 missed defects) |
| 8 | `10.1038/s41467-026-78143-2` | nature_communications | 22 | Clean (0 missed defects) |
| 9 | `10.1038/s41467-026-78768-3` | nature_communications | 13 | Clean (0 missed defects) |
| 10 | `10.1038/s41467-026-79283-1` | nature_communications | 1 | Clean (0 missed defects) |
| 11 | `10.1038/s41746-025-02298-3` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 12 | `10.1038/s41746-025-02319-1` | npj_digital_medicine | 8 | Clean (0 missed defects) |
| 13 | `10.1038/s41746-026-02390-2` | npj_digital_medicine | 9 | Clean (0 missed defects) |
| 14 | `10.1038/s41746-026-02474-z` | npj_digital_medicine | 15 | Clean (0 missed defects) |
| 15 | `10.1038/s41746-026-02487-8` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 16 | `10.1038/s41746-026-02548-y` | npj_digital_medicine | 26 | Clean (0 missed defects) |
| 17 | `10.1038/s41746-026-02564-y` | npj_digital_medicine | 4 | Clean (0 missed defects) |
| 18 | `10.1038/s41746-026-02598-2` | npj_digital_medicine | 8 | Clean (0 missed defects) |
| 19 | `10.1038/s41746-026-02600-x` | npj_digital_medicine | 6 | Clean (0 missed defects) |
| 20 | `10.1038/s41746-026-02651-0` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 21 | `10.1038/s41746-026-02722-2` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 22 | `10.1038/s41746-026-02735-x` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 23 | `10.1038/s41746-026-02774-4` | npj_digital_medicine | 3 | Clean (0 missed defects) |
| 24 | `10.1038/s41746-026-02868-z` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 25 | `10.1038/s41746-026-02879-w` | npj_digital_medicine | 3 | Clean (0 missed defects) |
| 26 | `10.1038/s41746-026-02915-9` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 27 | `10.1038/s41746-026-03010-9` | npj_digital_medicine | 40 | Clean (0 missed defects) |
| 28 | `10.1038/s41746-026-03065-8` | npj_digital_medicine | 32 | Clean (0 missed defects) |
| 29 | `10.1038/s41746-026-03079-2` | npj_digital_medicine | 20 | Clean (0 missed defects) |
| 30 | `10.1371/journal.pmed.1004868` | plos_medicine | 1 | Clean (0 missed defects) |
| 31 | `10.1038/s41467-026-68579-x` | nature_communications | 3 | Clean (0 additional defects within accessible artifacts) |
| 32 | `10.1038/s41467-026-68972-6` | nature_communications | 15 | Clean (0 additional defects within accessible artifacts) |
| 33 | `10.1038/s41467-026-72347-2` | nature_communications | 36 | Clean (0 additional defects within accessible artifacts) |
| 34 | `10.1038/s41467-026-72436-2` | nature_communications | 42 | Clean (0 additional defects within accessible artifacts) |
| 35 | `10.1038/s41467-026-74122-9` | nature_communications | 2 | Clean (0 additional defects within accessible artifacts) |
| 36 | `10.1038/s41467-026-74472-4` | nature_communications | 1 | Clean (zip-only esm_1.zip extracted on secondary check: 4 Source Data XLSX files, 0 alerts) |
| 37 | `10.1038/s41467-026-75642-0` | nature_communications | 1 | Clean (0 additional defects within accessible artifacts) |
| 38 | `10.1038/s41467-026-77038-6` | nature_communications | 39 | Clean (0 additional defects within accessible artifacts) |
| 39 | `10.1038/s41467-026-77183-y` | nature_communications | 43 | Clean (0 additional defects within accessible artifacts) |
| 40 | `10.1038/s41467-026-77355-w` | nature_communications | 1 | Clean (0 additional defects within accessible artifacts) |
| 41 | `10.1038/s41467-026-78899-7` | nature_communications | 4 | Clean (0 additional defects within accessible artifacts) |
| 42 | `10.1038/s41746-026-02520-w` | npj_digital_medicine | 40 | Clean (0 additional defects within accessible artifacts) |
| 43 | `10.1038/s41746-026-02634-1` | npj_digital_medicine | 2 | Clean (0 additional defects within accessible artifacts) |
| 44 | `10.1038/s41746-026-02697-0` | npj_digital_medicine | 40 | Clean (0 additional defects within accessible artifacts) |
| 45 | `10.1038/s41746-026-02708-0` | npj_digital_medicine | 2 | Clean (0 additional defects within accessible artifacts) |
| 46 | `10.1038/s41746-026-02754-8` | npj_digital_medicine | 1 | Clean (0 additional defects within accessible artifacts) |
| 47 | `10.1038/s41746-026-02782-4` | npj_digital_medicine | 7 | Clean (verified RandomOverSampler inside ImbPipeline + continuous Vickers–Elkin DCA) |
| 48 | `10.1038/s41746-026-02834-9` | npj_digital_medicine | 8 | Clean (0 additional defects within accessible artifacts) |
| 49 | `10.1038/s41746-026-02926-6` | npj_digital_medicine | 29 | Clean (0 additional defects within accessible artifacts) |
| 50 | `10.1038/s41746-026-03150-y` | npj_digital_medicine | 1 | Clean (zip-only esm_1.zip extracted on secondary check: 8 Python scripts, 0 alerts) |

---

## Supplementary Note S1. Detailed Forensic Derivations for the Four 2026 Lifecycle Reference Cases

### S1.1 Stage 1 — Multi-Omics Mirror Cloning & Non-Deduplicated `.isin()` Root Cause (Chen et al., 10.1038/s41746-026-02991-x)
- **Official Artifact Verification:** `MOESM4_ESM.xlsx` SHA-256 verified identically across independent from-scratch audit (`cases/case1_02991x/independent_verify_02991x.py`).
- **Exact 17-Digit Test Clone Proportions:** Early Fusion 62/96 (64.58%), DNA Methylation 171/249 (68.67%) (including 1 contradictory-label profile carrying both y=0 and y=1 in both splits), Proteomics 122/183 (66.67%), Transcriptome 72/114 (63.16%).
- **Exact Multiplicity Parity & Root-Cause Chi-Square Test:** Across all 4 modalities, 142/142 (100.0%) shared profiles satisfy exact pre-ADASYN parity ($k_{\text{train,pre}} = k_{\text{test}}$, up to (24, 24)), 0/216 multiplicity-2 profiles split (1, 1) ($P \le 10^{-14.84}$ to $10^{-34.13}$), and observed (2,0):(2,2):(0,2) counts match an 80/20 `train_test_split` on non-deduplicated `submitter_id`s followed by `.isin()` subsetting ($\chi^2 = 0.055\text{–}1.249$, $\text{df}=2$, all $P > 0.53$).
- **Unseen Discrimination & Confusion-Matrix Bounds:** Independent pure-NumPy Random Forest reproduces Table 1 on full leaky test sets (0.949 / 0.948 / 0.953 / 0.922 vs. published 0.939 / 0.955 / 0.962 / 0.928), scores 1.000 (0.9999 Methylation) on leaked clones, and collapses on truly unseen unique test patients (n = 32, 75, 58, 40) to 0.719 [0.502, 0.918], 0.491 [0.353, 0.630], 0.521 [0.324, 0.714], and 0.617 [0.416, 0.822] (L2 logistic regression: 0.525 / 0.518 / 0.647 / 0.577). Main-text Table 1 omitted Specificity and Table 2 transposed TN/FP; decomposing Supplementary Figs. 1–4 yields memorization-implied unseen specificity (supported by flat 1.000 training AUC in Supp Fig. 5) of 25.0% (3/12), 12.5% (3/24), 22.2% (4/18), and 21.4% (3/14) (monotonicity upper bounds $\le 50.0\% / 72.2\% / 79.4\% / 45.0\%$).

### S1.2 Stage 2 — Clinical Evaluation Degradation (Lin et al., 10.1038/s41746-026-02470-3)
- **Binarized Baseline:** Continuous 0–9 CHA~2~DS~2~-VASc dichotomized into {0, 1} (`cha2ds2_pred`) before `roc_auc_score`, `roc_curve`, and DeLong testing (NB04/NB06), collapsing AUROC to exact balanced accuracy (0.674 / 0.614 / 0.621).
- **Hardcoded Flat DCA:** `net_benefit_binary_rule()` (NB07) hardcodes `threshold = 0.2` and tiles constant net benefit (32.9 / 1,000 EV1; 52.5 / 1,000 EV2) across $p_t \in [0.01, 0.99]$, understating true Vickers–Elkin net benefit 3.4×–5.0× at atrial fibrillation anticoagulation thresholds ($p_t = 0.02\text{–}0.10$: true 163.8 / 145.4 / 112.1 vs. 32.9 EV1) and reversing sign above $p_t = 0.235$.
- **Additional Evaluation Defects:** Binary {0,1} predictions yielded invalid Brier (0.486 = 1 - \text{Acc}) and frozen NRI (0.4903 across $t = 0.2\text{–}0.6$); Figure 3f swapped Cohort 1/2 calibration axes (NB06 C78); Figure 5 stratified baseline Kaplan–Meier survival by post-baseline 3- and 5-year NOAC exposure (`NOAC_3y/5y`, immortal-time bias).

### S1.3 Stage 3 — Statistical Inference Failures Surviving Transparent Peer Review (Jin et al., 10.1038/s41467-026-78947-2)
- **Summary Within 2026 Cross-Journal Census (Non-Overlapping With Dedicated Parallel Correspondence):** Official `MOESM4_ESM.xlsx` and `MOESM3_ESM.pdf` document 111/111 (100.0%) fixed ±0.003 CIs for the three revision-added models (CONCH-1.5, Virchow, Hibou-L), median 27.6× (range 18.8×–55.0×, matching $\sqrt{1000} = 31.6$) CI shrinkage between Figure 6 and Figures 2–4 across 37 matched TCGA tasks (plus zero-width [0.992, 0.992] on GBMLGG-IDH1), and bootstrap-pseudoreplication Wilcoxon tests down to $P = 1.46 \times 10^{-161}$ ($\Delta\text{F1} = +0.001 \rightarrow P = 1.97 \times 10^{-159}$).
- **Null Calibration & Patient-Level Re-Evaluation:** Monte Carlo null simulation ($B = 1{,}000$, 500 trials, $\Delta = 0$) confirms 48.2% false-positive rate at $P < 0.05$ (46.6% at $P < 0.001$) and 22.4/47 (47.7%) pure-null tasks falsely passing BH-FDR $q < 0.05$. Evaluated properly at the patient level, 38/47 (80.9%) primary comparisons have $P > 0.05$ at $\rho = 0.50$ (1/47 passing BH $q < 0.05$; 10/47 even at $\rho = 0.80$).

### S1.4 Stage 4 — Live Bedside Risk Calculator Safety Failure (Zhu et al., 10.1371/journal.pmed.1005078)
- **`code.R` Line 648 Sign Destruction:** `weights <- abs(coef_vec)` removes negative signs from all 6 protective Cox predictors (PIR3, ALB, Education3, EGFR, LYM, SCR), dropping all-cause C-index 0.8723 → 0.8384 (5-fold OOF 0.8702), CVD C-index 0.8917 → 0.8574 (matching Age alone 0.8569), and destroying 72.51% of age+sex-adjusted LRT $\chi^2$ (339.77 → 93.40). Stage 4 CKD (eGFR 100→20) with hypoalbuminemia (ALB 4.5→2.5) paradoxically reduces published CKMAI by ~25–26 age-equivalent years.
- **`CKMAIcalc::predict_risks()` Sex-Encoding Contradiction:** `CKMAI_nomogram_model.rds` uses Sex ∈ {1=M, 2=F} ($\beta = -0.4688$), whereas `CKMAI_nomogram_model_cvd.rds` ($\beta = +0.6621$) and `CKM_high_logistic.rds` ($\beta = +0.8605$) use Sex ∈ {1=M, 0=F}, yet `predict_risks.R` passes raw Sex ∈ {1,2} to all three. Across 3,293 NHANES women (`S1_Data.xlsx`, $n = 6{,}896$), 858/3,293 (26.06%) receive impossible 10y CVD > 10y all-cause mortality (vs. 0/3,293 harmonized and 0/3,603 males); female mean 10y CVD risk is inflated 2.84× (4.25% → 12.08%, median individual 3.72×), female `CKM_high` risk is inflated 2.80× (11.48% → 32.11%), and overall `CKM_high` AUROC drops 0.8383 → 0.7892.
