# Primary-Source Verification Record (`ClinAudit` 2026 Empirical Audit)

**Author:** Xingyu Zhang (ORCID: `0009-0005-6181-7465`)  
**Scope:** Verbatim primary-source code lines, publisher XML quotations, triage-control justifications, and negative-control spot checks supporting the 2026 cross-journal audit and late-2025 out-of-time validation. All artifact paths below are repository-relative (third-party publisher PDFs/XLSX files are fetched directly from official DOIs via the archive scripts and are not redistributed).

---

## Part 1: Prospective Blind 2026 Discoveries (`ClinAudit v0.1.0`)

### 1A. Turki et al. (*Nature Communications*, `10.1038/s41467-026-70584-z`, Mar 20, 2026) — 44.7× Bootstrap-SE CI Collapse
- **Source Script:** `harvest/artifacts_n3/10.1038_s41467-026-70584-z/gh_aipal_validation__ml__test.py`
- **Source Data:** `harvest/artifacts_n3/10.1038_s41467-026-70584-z/esm_1.xlsx` (`Suppl. Data 1`, `Suppl. Data 4`)
- **Verbatim Call Chain (Lines 204–213, 251–259, 314–319):**
```python
# Lines 204-213: compute_statistical_metrics()
mean_val = np.nanmean(values)
std_dev = np.nanstd(values)
ci_width = norm.ppf((1 + confidence_level) / 2) * (
    std_dev / np.sqrt(iterations)
)
final_results[cat][metric] = (
    mean_val,
    mean_val - ci_width,
    mean_val + ci_width,
)

# Lines 251-259: bootstrap_metrics()
for _ in range(iterations):
    sampled_data = resample(data)
    metrics = self.calculate_metrics(sampled_data, cutoff_type)
    ...
results = self.compute_statistical_metrics(results, iterations, confidence_level)

# Lines 314-319: run_evaluation()
for cutoff_type, cutoff_metric in cutoff_dict.items():
    results = evaluator.bootstrap_metrics(
        cutoff_type, cutoff_metric,
        iterations=2000 if not config["debug"] else 1,
    )
```
- **Published Propagation:** Dividing bootstrap SD by `np.sqrt(2000) = 44.721` narrows 95% CIs 44.7-fold directly into `Suppl. Data 1` and `Suppl. Data 4` (`166+` zero-width CIs) and published text (`0.82 [0.82–0.82]`, `0.95 [0.95–0.95]`). Zero prior corrections recorded on Crossref as of November 25, 2026.

---

### 1B. Axelsson et al. (*PLOS Medicine*, `10.1371/journal.pmed.1005190`, Aug 27, 2026) — Live Binarized Calibration & Uncensored/Survivor-Dropping C-Index
- **Source Script (1 of 5 identical XGBoost evaluation scripts):** `harvest/artifacts_n3/10.1371_journal.pmed.1005190/gh_PYTHON__Development__XGBoost__Evaluation__MultivariableXGBoost_testing_best_unique_35f.py`
- **Verbatim Execution Chain (Lines 37–56, 67–92, 157–177):**
```python
# Lines 37-56: Hard {0,1} binarization fed to linear regression & 10-bin calibration curve
def calculate_calibration(y_true, y_pred, years, output_dir):
    for year in years:
        days = year * 365
        y_pred_binary = np.array([1 if time < days else 0 for time in y_pred])
        y_true_binary = np.array([1 if time < days else 0 for time in y_true])
        cil = np.mean(y_pred_binary) - np.mean(y_true_binary)
        slope, intercept, r_value, p_value, std_err = linregress(y_pred_binary, y_true_binary)
        display = CalibrationDisplay.from_predictions(y_true_binary, y_pred_binary, n_bins=10, ax=ax, name=f"{year} Year(s)")

# Lines 67-82: Drops all survivors with y_true >= days AND omits event_observed
def calculate_concordance_index(y_true, y_pred, years, output_dir, n_bootstraps=1000, alpha=0.05):
    for year in years:
        days = year * 365
        mask = y_true < days
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        if len(y_true_filtered) > 0:
            c_index = concordance_index(y_true_filtered, y_pred_filtered)

# Lines 157-177 inside main(): Direct invocation + uncensored overall C-index right above censored KM
ci = concordance_index(y_true, y_pred)                           # L158: omits event_observed
kmf = KaplanMeierFitter()
kmf.fit(y_true, event_observed=(y_upper == y_lower))             # L165: censoring indicator is available
calculate_calibration(y_true, y_pred, years=range(1, 6), output_dir=output_dir)       # L173
calculate_concordance_index(y_true, y_pred, years=range(1, 6), output_dir=output_dir) # L176
```
- **Published Context:** The article notes that `"The XGBoost model did not allow for individual probability calculations and could therefore not be calibrated."` Nonetheless, all five official XGBoost evaluation scripts execute this binarized calibration routine directly inside `main()` alongside the uncensored and survivor-dropping C-index calculations. Zero prior corrections recorded on Crossref as of November 25, 2026.

---

## Part 2: Out-of-Time Late-2025 External Validation Discovery (Sep–Dec 2025, Frozen `v0.2.0`, `46` Downloaded / `45` Parsed, `4/45 = 8.9%` Parsed / `4/46 = 8.7%` Downloaded Flagged)

### 2A. Xiao et al. (*Nature Communications*, `10.1038/s41467-025-67656-x`, Dec 24, 2025, `PMC12830939`) — Constant-Seed Bootstrap Collapsing Published SDs to IEEE-754 Machine Epsilon (`1.11–4.44 × 10⁻¹⁶`)
- **Detection Path:** Frozen `ClinAudit v0.2.0` flagged `EVAL-004` (`rus.fit_resample(x_test/x_test_male/x_test_female)`) at lines 150, 324, and 377 of `UKB_by_sex_and_undersampling.py` (`4/46 = 8.70%` downloaded alert rate across Sep–Dec 2025; the other 3 flagged articles—`63846-9` vertically stacked Cox/LogRank/SHAP sheet, `02189-7` GSVA correlation sheet, and `66483-4` GO enrichment sheet—were schema false positives). Review of the flagged block confirmed that balanced test undersampling was disclosed as a sensitivity analysis, while revealing the adjacent constant-seed bootstrap loop across all 11 bootstrap blocks in `UKB_by_sex_and_undersampling.py` and `external_validation.py`.
- **Source Scripts:**
  - `harvest/external_2025/artifacts/10.1038_s41467-025-67656-x/gh_UKB_by_sex_and_undersampling.py` (L9, L149–178, L234, L283, L324–366, L377–416)
  - `harvest/external_2025/artifacts/10.1038_s41467-025-67656-x/gh_external_validation.py` (L9, L219–251, L271–303, L341–373, L393–425, L465–497, L517–549)
- **Verbatim Code (L9, L149–169):**
```python
random_state = 1996
...
rus = RandomUnderSampler(random_state=random_state)
X_resampled, y_resampled = rus.fit_resample(x_test, y_test)
num_iterations = 100
for _ in range(num_iterations):
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled, random_state=random_state)
    y_pred = clf.predict_proba(X_bootstrap)[:, 1]
    ...
std_auc = np.std(auc_scores)
print(f'AUC: {auc:.3f} +/- {std_auc:.3e}')
```
- **Verbatim Published Main-Text XML Quotations (`PMC12830939` — 25 collapsed uncertainty estimates: 21 at `1.11–4.44 × 10⁻¹⁶` and 4 at `2.06 × 10⁻²⁹⁶`; 0 corrections on Crossref):**
  - `"When case to control ratio is 1:1, the AUC stays the same at 0.95 (SD = 2.22 × 10⁻¹⁶) (F1 = 0.99, SD = 1.11 × 10⁻¹⁶; AP = 0.97, SD = 2.22 × 10⁻¹⁶; BA = 0.89, SD = 3.33 × 10⁻¹⁶)."`
  - `"externally validated in CKB, achieving an AUC of 0.91 (SD = 2.22 × 10⁻¹⁶; Fig. 2a) (F1 = 0.93, SD = 1.11 × 10⁻¹⁶; AP = 0.87, SD = 1.11 × 10⁻¹⁶; BA = 0.79, SD = 2.22 × 10⁻¹⁶)"`
  - `"higher performance in females compared to males, with an AUC of 0.96 (SD = 1.11 × 10⁻¹⁶) and 0.94 (SD = 2.06 × 10⁻²⁹⁶), respectively."`

---

## Part 3: Four Rule-Motivating 2026 Lifecycle Reference Cases (Independently Reproduced)

- **3A. Stage 1 — Chen et al. (*npj Digital Medicine*, `10.1038/s41746-026-02991-x`):** Detected blind by `v0.1.0` (`LEAK-001`/`LEAK-003`). Independent verification archive: `cases/case1_02991x/VERIFICATION_SUMMARY.md` and `cases/case1_02991x/independent_verification_report.json` (SHA-256-verified `MOESM4_ESM.xlsx`: `63.2%–68.7%` exact 17-digit test clones; `142/142` `(k, k)` mirror parity; `.isin()` split $\chi^2 = 0.055$–$1.249$, $P > 0.53$; unseen RF AUC `0.719 / 0.491 / 0.521 / 0.617`).
- **3B. Stage 2 — Lin et al. (*npj Digital Medicine*, `10.1038/s41746-026-02470-3`):** Official notebooks `harvest/artifacts_n3/10.1038_s41746-026-02470-3/gh_04_LR_model_external_validation.ipynb` and `gh_07_*` (`cha2ds2_pred` `{0,1}` binarization, hardcoded `threshold = 0.2` flat DCA `32.9/1000` vs. true `163.8/1000` at $p_t = 0.02$, Fig. 5 `NOAC_3y/5y` immortal-time stratification).
- **3C. Stage 3 — Jin et al. (*Nature Communications*, `10.1038/s41467-026-78947-2`):** Official Source Data `harvest/artifacts_n3/10.1038_s41467-026-78947-2/esm_1.xlsx` (`111/111` fixed `±0.003` CIs for revision-added models; `27.6×` Fig. 6 vs. Figs. 2–4 width ratio; bootstrap-pseudoreplication Wilcoxon $P$ to `1.46e-161`).
- **3D. Stage 4 — Zhu et al. (*PLOS Medicine*, `10.1371/journal.pmed.1005078`):** Reproduction script and summary `plos_safety_package/verify_ckmaicalc_safety.py`, `plos_safety_package/results/ckmaicalc_safety_summary.json`, and `harvest/artifacts_n3/10.1371_journal.pmed.1005078/gh_code.R` (L648 `weights <- abs(coef_vec)` destroying `72.51%` LRT $\chi^2$; cross-model `{1,2}` vs. `{0,1}` `Sex` conflict causing `858/3,293 = 26.06%` women to receive impossible `10y CVD > 10y all-cause mortality`).

---

## Part 4: Human-in-the-Loop Triage Controls & Minor Reporting Error (Excluded from Core Defects)

- **4A. Triage Control 1 (Explicitly Disclosed Limitation) — Fletcher-Lloyd et al. (*npj Digital Medicine*, `10.1038/s41746-026-02964-0`):**
  - Source notebook: `harvest/artifacts_n3/10.1038_s41746-026-02964-0/gh_age_estimation__age_estimation.ipynb` (L76: `test_data[args.x_cols] = scaler.fit_transform(test_data[args.x_cols]) # a new scaler should be fit to each new dataset`).
  - Full-text Discussion explicitly discloses: `"A notable limitation of the current study is independently scaling external datasets for age estimation, which was necessitated by changes in hardware and firmware."` Excluded from undisclosed defects.
- **4B. Triage Control 2 (Validation-Set Naming Ambiguity) — Wang et al. (*Nature Communications*, `10.1038/s41467-026-74209-3`, *T2Pdecoder*):**
  - Source script: `harvest/artifacts_n3/10.1038_s41467-026-74209-3/gh_src__T2Pdecoder_VAE_train.py` (L155–159: `if MSE_loss_t.item() < min_test_loss: torch.save(...)`).
  - Full-text Methods verification showed that for 3/4 scripts (`CLIP_pre_train.py`, `CLIP_fine_tune.py`, `T2Pdecoder_VAE_train.py`), the `test_ratio=0.1` split named `x_test` functioned as an internal validation split paired with independent external `"Unseen"` (`n=5,608`) and `TSKGA` test cohorts. Reclassified as a naming-ambiguity triage control demonstrating why static `EVAL-006` flags require Methods verification.
- **4C. Minor Reporting Error — Zhou et al. (*Nature Communications*, `10.1038/s41467-026-70077-z`):**
  - Source Data: `harvest/artifacts_n3/10.1038_s41467-026-70077-z/esm_1.xlsx` (`Supplementary Data 7`, row 23: inverted 95% CI `0.996 (0.995, 0.887)` where upper bound `0.887 <` lower bound `0.995`; `Supplementary Data 3 & 6`: zero-width `0.988 (0.988, 0.988)`).

---

## Part 5: Spot-Check Summary of the 50 Random Unflagged Controls (`seed = 2026` & `2026050`; `0/50` Additional Defects Within Accessible Artifacts; NPV = `100.0%`, FOR = `0.0%`, 95% Upper Bound = `7.11%`; `39/50` Code, `25/50` Tables, `2/50` Zip-Only Verified Clean on Extraction)

Full 50-paper control records are archived in `harvest/processed/negative_control_sample_30.json` and `harvest/processed/negative_control_sample_50_raw.json`. High-risk oversampling and DCA spot checks verified directly from source lines:
- **5A. `10.1038/s41467-025-68198-y` (Trauma mortality ML):** `harvest/artifacts_n3/10.1038_s41467-025-68198-y/gh_train.py` lines 17–18 apply `SMOTE(random_state=77).fit_resample(X_train, y_train)` strictly within the training split (CLEAN).
- **5B. `10.1038/s41467-026-73021-3` (Pediatric cardiology ML):** `harvest/artifacts_n3/10.1038_s41467-026-73021-3/gh_models__GOSH__GOSH_traditional_models.ipynb` lines 193, 207, 221, 233, and 465 wrap `("oversample", RandomOverSampler(random_state=42))` strictly inside `imblearn.pipeline.Pipeline` during cross-validation (CLEAN).
- **5C. `10.1038/s41746-026-02782-4` (Oversampling + DCA control #47):** `harvest/artifacts_n3/10.1038_s41746-026-02782-4/gh_01_train.py` lines 96–100 wrap `('sampler', RandomOverSampler(random_state=42))` inside `ImbPipeline` within `GridSearchCV.fit(X_train, y_train)`, and `gh_05_plot_dca_curve.py` lines 52–90 compute continuous threshold-varying Vickers–Elkin net benefit (CLEAN).
- **5D. Secondary zip-only extractions:** `10.1038/s41467-026-74472-4` (4 Source Data `.xlsx` files, 0 alerts) and `10.1038/s41746-026-03150-y` (8 Python scripts, 0 alerts).
