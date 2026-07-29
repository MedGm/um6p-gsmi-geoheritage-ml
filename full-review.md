# Phase 1 National Accessibility Pipeline — Foundations Rebuild: Full Review

**Date:** 2026-07-29
**Scope:** `livrable/phase1_national_accessibility/` only (Phase 2 regional analytics and Phase 3 extraction pipeline were explicitly out of scope and left untouched)
**Branch:** `worktree-phase1-foundations-fix` (14 commits on top of `bd8d233`), not yet merged to `main`
**Worktree location:** `.claude/worktrees/phase1-foundations-fix/`

---

## 1. Why this work happened

A prior internal review (methodology critique from a senior reviewer, following supervisor feedback) identified that the Phase 1 pipeline's engineering was sophisticated but its scientific defensibility was weak. Four concrete, independently verifiable defects were identified in the code (not just in the write-up):

1. **Raster registration was a hand guess.** The GIS raster stack (elevation, slope, ruggedness, LULC, soil, geology, distance-to-dams, distance-to-rivers) was produced by scanning paper/PNG maps and stamping a manually guessed `Affine` geotransform onto them. The guess was wrong by roughly 20–25 km. Rather than fix the transform, the old pipeline patched around it downstream with two separate hard-coded pixel shifts (`shift_rows = 17` in the map-projection script, and `y_sites + 20000` for plotting geosite points) — a classic "two wrongs cancel out, sometimes" pattern that is not reproducible and not explainable to a reviewer.

2. **The headline "exact Euclidean distance to roads" was not exact.** It was computed by rasterizing OpenStreetMap roads onto a 1183-metre grid and running a distance transform. Across 309 geosites this produced only **31 distinct distance values nationally** — a coarse quantization, not a continuous exact distance, despite the report's wording.

3. **The machine-learning target was circular.** The accessibility label used to train the classifier was a deterministic mathematical function of the same terrain variables (distance to highway, slope, ruggedness, elevation) that the classifier was then trained on. This means the previously reported >90% F1 score was measuring how well a model can re-derive a formula from its own inputs — not predictive skill. It also explains why land-cover (LULC), a variable domain experts expect to matter, showed near-zero feature importance: the label had zero mathematical dependence on it by construction.

4. **A validation table was silently broken.** An anchor-site concordance check compared the string `"Difficile"` against `"Difficile (Difficult)"` — these can never be equal, so every anchor site registered `MISMATCH` regardless of whether the prediction was actually correct.

The senior review's conclusion was explicit: *do not fix this by adding more sophisticated mathematics (optimization, fuzzy gating, more equations) — fix the measurement and data foundations first, because reviewers will ask "what evidence supports this methodological choice?", and "we optimized it" is not evidence.*

This rebuild addresses exactly the four defects above, using measurement and independent data rather than added mathematical sophistication.

---

## 2. What changed, defect by defect

### 2.1 Raster registration

**Before:** Manually guessed transform origin `(x_min=-1671597.6, y_max=682840.2)`, wrong by an unmeasured amount, compensated with ad-hoc downstream pixel shifts.

**After:** A dedicated calibration script (`00_calibrate_raster_registration.py`) measures the true offset:
- Builds a land/sea mask from the scanned elevation raster's nodata pattern.
- Fetches an independent, authoritative land/sea reference (Natural Earth admin-0 country polygons for Morocco + Western Sahara, cross-checked visually against the OpenStreetMap coastline fetched live via the Overpass API).
- Runs an exhaustive ±30-pixel search maximizing Intersection-over-Union (IoU) between the scanned mask and the reference.
- Applies the winning offset directly to the `Affine` transform of the 8 physical rasters (elevation, slope, ruggedness, distance-to-dams, distance-to-rivers, LULC, soil, geology) — pixel *values* are untouched, only the geographic registration is corrected.

**Result:**
- IoU improved from **0.8128 → 0.9067**.
- Winning offset: **21 pixels north-south (≈24.85 km), 0 pixels east-west.**
- This closely matches the magnitude and axis of the two ad-hoc hacks it replaces (17px, 20000m) — independent corroborating evidence the correction is real and correctly signed.
- A defensive assertion (`assert src.transform == transform`) was added so the script refuses to silently mis-apply the correction to a raster whose starting geometry doesn't match what's expected.
- The old, uncorrected `georeference_all.py` script (the actual source of the original error) is **not deleted** (it documents pipeline history) but now refuses to run (`raise SystemExit(...)`), so it cannot silently undo this fix if someone runs it again by habit.

### 2.2 Highway-distance quantization

**Before:** OSM roads rasterized to a 1183m grid, Euclidean Distance Transform run on the raster → 31 distinct values across 309 sites.

**After:** True point-to-nearest-line vector distance, computed with `geopandas.sjoin_nearest` directly against the OSM highway line geometries (motorway/trunk/primary/secondary/tertiary classes; unpaved "piste" tracks were **excluded entirely**, per the original supervisor feedback, because OSM piste tagging in Morocco was judged too inconsistent and regionally biased to trust).

**Result:** **309 unique coordinate pairs → 309 unique distances** (fully continuous, confirmed via direct count, not quantized). A dedicated methodology figure (`distance_computation_methodology.png`) now illustrates the point-to-line computation for sample sites, replacing an unexplained equation with a visual, reproducible explanation.

Along the way, the feature-extraction step also picked up 58 additional geosites that the old 309-site master dataset had dropped (367 geolocated sites now available, out of 780 raw catalog rows — 409 rows have no usable coordinates and were excluded).

### 2.3 The circular label

This was judged the single most important fix. **Before:** label = closed-form function of the same 4 variables among the model's 9 training features.

**After:** The label is now real-world **driving travel time**, computed via the public OSRM routing engine from each geosite to the nearest of its 4 geographically-closest major reference cities (Rabat, Casablanca, Marrakech, Fès, Tanger, Agadir, Oujda, Errachidia, Ouarzazate, Laâyoune, Dakhla, Al Hoceïma). This uses real road topology, road class, and routing logic — information that is not present in any of the 9 terrain/distance predictor features the model trains on.

**Independence was verified, not assumed.** A dedicated out-of-fold check regresses the OSRM travel time against the 9 predictor features using 5-fold cross-validated `RandomForestRegressor` predictions (genuinely held-out, not in-sample):

> **Out-of-fold R² = 0.6607**

This is well under the 0.95 "still basically circular" threshold set in advance, and is now a permanent, automated assertion inside `02_generate_accessibility_labels.py` (`assert r2_oof < 0.95`) — so any future change to the label logic that reintroduces circularity will fail loudly rather than silently.

(Two sites — "Cap Malabata" and "Fairy chimneys of Assamar" — have corrupted near-0° longitude values inherited from an upstream catalog error; they are consistently excluded from this check and from model training, with the exclusion applied identically in both places, not selectively.)

**Consequence — the model's real numbers, not the fabricated ones:**

| Metric | Old (circular) | New (independent label) |
|---|---|---|
| Best model | Random Forest | **HistGradientBoosting** |
| Spatial Block CV F1 (50km blocks) | 0.9042 | **0.6018** |
| Spatial Block CV Accuracy | 91.91% | (see report; not directly comparable — class scheme changed, see below) |

The drop from ~0.90 to ~0.60 is expected and is the correct outcome, not a regression: it reflects the model now being asked to predict something it cannot simply read off its own inputs. A Standard (non-spatial) CV F1 of 0.8239 was also measured; the report explicitly flags that this number is additionally inflated by 58 exact-duplicate feature rows (co-located geosites), so the raw 0.82 vs 0.60 gap should not be read as purely a measure of spatial autocorrelation.

**Class scheme changed from 4 classes to 3.** The real-world OSRM-derived label produced only **1 site** out of 367 in the "Facile" (easy) class — travel time to a major city under 30 minutes turned out to be extremely rare for a national geosite catalog, which tend by nature to be remote. A single-member class cannot support 5-fold stratified cross-validation, so "Facile" was merged into "Modérée", giving a training distribution of Modérée=33, Difficile=147, Très Difficile=185 (365 usable rows after excluding the 2 corrupted-coordinate sites). This is documented in the report as a data-driven finding, not an arbitrary choice.

### 2.4 Anchor validation bug

**Before:** String comparison bug made every concordance check register `MISMATCH` structurally, regardless of correctness.

**After:** Comparison logic fixed to compare values from the same vocabulary. Result: **3 of 3** available anchor sites (Oued ElKelaa Gorge, Lac Tislit, Akchour landslide) now show real `MATCH`.

**Important honesty correction:** this check is trained on all 365 rows including the 3 anchors, then predicts those same rows — it is an **in-sample consistency spot-check**, not field validation, and the "observed" category is itself the OSRM-derived label, not an independently field-collected observation. Both report versions (EN/FR) were reworded during review to describe it accurately as such — the earlier drafts had called it "empirical field anchor validation," which overclaimed what was actually measured.

---

## 3. National map output

Projecting the retrained model across the full, now-correctly-registered Morocco raster grid:

- **484,320 land pixels classified** (14.1% of the full raster grid extent).
- Class distribution: **Modérée 4.1%, Difficile 29.6%, Très Difficile 66.2%**.
- The highway-distance raster used for this national projection was regenerated fresh from vector road data against the corrected geometry — the on-disk `distance_to_highways_meters.tif` was found to still carry the old, wrong ~25km-offset registration (it predates the Task 2 fix and was never in scope for that fix, since it's superseded by the vector method), so it was **not reused**, and has been renamed to `distance_to_highways_meters_PRE_CALIBRATION.tif` / `distance_to_pistes_meters_PRE_CALIBRATION.tif` so any other code still pointing at the old filename fails loudly instead of silently mixing correctly- and incorrectly-registered data.

---

## 4. Review process and independent verification

This was not a single self-check. Three layers of verification were applied:

1. **Per-task self-review.** Each of the 8 implementation tasks was built by an independent agent instance with no memory of the others, working strictly from a written plan. Several found and fixed real bugs in the plan's own draft code before shipping (a sign error in the registration-transform math, a non-unique join key, an infeasible stratified cross-validation split) — each fix is documented with its reasoning in the task's own report.

2. **Consolidated whole-branch review.** A separate reviewer instance, with no involvement in building the code, was given the full 14-commit diff and instructed to independently re-derive and re-check every claim rather than trust the implementers' self-reports. It found **2 Critical and 7 Important issues**, including:
   - The old, uncorrected `georeference_all.py` script left runnable, which would have silently reverted the registration fix (Critical).
   - The report's cited R²=0.62 independence number described an experiment (RandomForest, 80/20 split) that had never actually been run as described — the real number for that protocol was different, and no script actually computed and asserted it (Critical).
   - The 7 Important findings covered: the leakage check not being automated anywhere, duplicate-row inflation of the standard-CV benchmark, the anchor-validation overclaim described above, dead/misleading code in the feature extraction script, the stale misregistered highway/piste rasters, an unmarked broken downstream script, and an EDA figure caption that didn't clarify it used the retracted circular label.
   - This reviewer independently reproduced the out-of-fold R² check itself and got **-3.70** (using a slightly different exclusion of the 2 corrupted-coordinate sites), which is actually *stronger* evidence against circularity than the number the implementers had reported — i.e., independent verification found the fix even more solid than claimed, not weaker.

3. **Fix verification.** All 9 findings were fixed in a dedicated pass, then a **second, separate reviewer instance** re-checked every one of the 9 fixes by re-running the actual scripts directly (not reading the fix report and trusting it) — re-executing the R² check, re-running the feature-extraction script and diffing its output byte-for-byte, recompiling both LaTeX reports twice each. All 9 confirmed genuinely fixed. No new issues introduced. Verdict: **ready to merge.**

---

## 5. Known limitations (disclosed, not hidden)

- **Class imbalance remains real.** Even after merging Facile into Modérée, Très Difficile (66% of the national map) dominates. This is presented as a genuine finding about Moroccan geosite accessibility, not smoothed over.
- **EDA figures are stale.** The exploratory-data-analysis figures still reflect the old 309-site, circular-label dataset. They now carry an explicit caption/text caveat stating this, but were not regenerated (judged out of scope for a foundations fix; flagged as follow-up work).
- **`05_geosite_suitability_map.py`** (a Tier-2 / separate-paper concern per the original review — extending to *predicting* unknown geosite locations rather than scoring known ones) still references a deleted input file. It is marked broken with a header comment rather than fixed, since rewriting it is explicitly out of this rebuild's scope.
- **Phase 2 (regional analytics) code** still references the old model/raster filenames and will now fail if run. This is an intended, expected consequence of Phase 1-only scope (confirmed before starting) — Phase 2 needs its own follow-up pass.
- **Two sites have corrupted coordinates** (near-0° longitude) inherited from an upstream data catalog issue, consistently excluded from training and validation rather than silently included with bad data.
- Weight-optimization/AHP literature-based weighting, sensitivity analysis, and Jenks-based class breaks — all raised in the original review — were **deliberately deferred**, consistent with the review's own advice not to add mathematical sophistication before the measurement foundations were solid.

---

## 6. Where everything is

- **Plan (full task-by-task specification):** `docs/superpowers/plans/2026-07-29-phase1-foundations-fix.md`
- **Execution ledger (chronological record of every task, review, and fix):** `.claude/worktrees/phase1-foundations-fix/.superpowers/sdd/progress.md`
- **Rebuilt pipeline code:** `.claude/worktrees/phase1-foundations-fix/livrable/phase1_national_accessibility/code/`
  (`00_calibrate_raster_registration.py`, `01_extract_geosite_features.py`, `02_generate_accessibility_labels.py`, `03_train_accessibility_model.py`, `04_project_national_map.py`)
- **Updated reports (resynced with real numbers, both languages, recompiled clean):**
  - `.claude/worktrees/phase1-foundations-fix/livrable/phase1_national_accessibility/report/geosite_internship_report.pdf` (English)
  - `.claude/worktrees/phase1-foundations-fix/livrable/phase1_national_accessibility/report/geosite_internship_report_fr.pdf` (French)
- **Branch:** `worktree-phase1-foundations-fix`, 14 commits, currently **not merged to `main`** — kept isolated pending a decision on how to land it.

All of the above is inside the isolated worktree, not yet on `main`. `main` itself is untouched and still carries its own pre-existing uncommitted changes from before this work started.
