# Superseded scripts

Every script here produced a real result at some point, but its output is no
longer read by anything, is not cited (by filename or by exact number) in
either paper (`report/geosite_ai_section_2026.tex`,
`report/geosite_ai_section_2026_paper2_regional.tex`), and is not cited in
`presentation/wrapup.tex`. Kept for provenance/history, not part of the
active pipeline. Classification was done by tracing every script's actual
file-write output against every other script and both papers' text
(exact-number matching, not guesswork) — see the project's cleanup-plan
discussion for the full trace if a supervisor wants to double-check a
specific one.

## `01_data_preparation_dead/` (23 files, formerly `code/`)

| File | What it did | What superseded it |
|---|---|---|
| `02_extract_terrain_road_features.py` | Terrain/road feature extraction for an early "geosite_v2" locality track | That track was abandoned; current features come from `01_data_preparation/`'s catalog chain instead |
| `03_generate_draft_labels.py` | Draft label generation, same abandoned track | Same |
| `04_train_baseline_model.py` | Baseline model on draft/unreviewed labels from the abandoned track | Superseded entirely by the reviewed-label pipeline in `02_modeling_and_analysis/` |
| `05_ablation_data_quality.py` | Data-quality ablation on the abandoned track's data | Input files no longer exist; topic not revisited elsewhere |
| `09_full_cluster_aware_retrain.py` | N=251 classification retrain | `02_modeling_and_analysis/` rebuilds fresh from the final catalog, never touches this output |
| `10_feature_ablation_loco.py` | Leave-one-covariate-out ablation, N=251 | Not cited; superseded by later feature-set work in `02_modeling_and_analysis/24` |
| `13_retrain_favorability_pilot.py` | Favorability model v2 | Superseded by v3 (`notebooks/`) then v4 (`08_train_favorability_model_v4.py`) |
| `14_render_favorability_map.py` | Favorability map v3, from local raster + old model | Superseded by `03_report_generation/make_favorability_v3.py` (live-DEM sampling + v4 model) |
| `15_merge_labels_and_retrain_newdb.py` | N=308 classification retrain | Same reason as `09` |
| `16_incremental_retrain_with_new_labels.py` | Further incremental classification retrain | Same reason as `09` |
| `17_full_retrain_from_regional_sources.py` | Last classification-retrain iteration before the modeling pipeline moved to `02_modeling_and_analysis/` | Same reason as `09` |
| `18_experiment_matrix.py` | First hyperparameter battery | Had a class-balance bug (RF-only), fixed in `19` then `20` |
| `19_final_battery.py` | Second hyperparameter battery | Fixed the `18` bug but had its own scope bug, fixed in `20` (kept, `01_data_preparation/10_model_family_battery_n733.py`) |
| `21_tiny_region_ablation.py` | Tests whether including sub-30-sample regions in the training pool hurts pooled LOGO accuracy | Distinct question from the "merge thin regions with a neighbor" finding actually published (`02_modeling_and_analysis/25`); not cited |
| `22_comprehensive_final.py` | Intended 5-model-family battery incl. CatBoost | Crashed before finishing (no output file exists); its early results survive only as a hand-transcribed log |
| `22b_h5_h6_resume.py` | Resume-only patch for `22`'s missing configs | Its N=733 numbers were hand-copied into an intermediate file, but the *published* model-family table uses the N=939 rerun (`02_modeling_and_analysis/21`) instead |
| `23_least_cost_path_friction.py` | Least-cost-path friction surface feature | Abandoned in favor of the OSM-graph routing-distance feature (`02_modeling_and_analysis/09b-13`), which *is* published |
| `24_mixed_effects_region.py` | Region-aware mixed-effects model | Not cited in either paper's current text (a `README.md` claim about this predates the current paper content) |
| `25_oof_calibration_conformal.py` | Calibration/conformal prediction, N=733 | Superseded by the N=939 rerun (`02_modeling_and_analysis/22`), which is what's cited |
| `27_lcp_feature_test.py` | Least-cost-path feature test | Part of the abandoned LCP chain, see `23` |
| `28_mcnemar_batch.py` | McNemar tests on `24`'s mixed-effects results | Not cited (downstream of an uncited script) |
| `29_lcp_bmk_slice.py` | LCP feature test, BMK slice | Part of the abandoned LCP chain, see `23` |
| `30_mixed_effects_leave_region_out.py` | Leave-region-out companion to `24` | Not cited, same reason as `24` |

Two related scripts are NOT here despite being superseded-in-part:
`01_consolidate_geosite_catalog.py` is kept in `01_data_preparation/` (renamed
`_catalog_helpers.py`) because its functions are still imported and executed
live by the catalog-ingestion scripts, even though its own direct run output
is historical. `06/07/08` (DEM/LULC/settlement-distance migrations) are kept
in `01_data_preparation/historical_one_shot_migrations/` — real steps whose
effect is permanently baked into the production catalog, just not rerunnable
today since their own source inputs no longer exist.

## `02_modeling_and_analysis_dead/` (2 files, formerly `data_audit/`)

| File | What it did | What superseded it |
|---|---|---|
| `09_osm_routing_distance.py` | First attempt at OSM routing-distance extraction | OOM-killed at national scale; rewritten as the subprocess-isolated `09b/09c/09d` worker/driver/retry chain (kept, live) before this version produced a final artifact |
| `11_routing_mcnemar_regional.py` | McNemar test of the routing-distance feature, region-restricted | Used the wrong CV protocol (LOGO-cluster instead of leave-region-out), giving a misleading null result — corrected in `12` (kept, live), which is what's actually cited |
