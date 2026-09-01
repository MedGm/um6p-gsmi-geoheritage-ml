<p align="center">
  <img src="report/um6p_figures/um6p_gsmi_logo.png" width="420" alt="UM6P GSMI">
</p>

# Geosite Accessibility Modeling — Morocco

Machine-learning assessment of physical accessibility for Morocco's geoheritage sites
(geosites), built at the Geology and Sustainable Mining Institute (GSMI), UM6P. The
catalog spans **1,667 geosites** across all twelve administrative regions, of which
**1,662** carry an independently-sourced, citation-traceable accessibility label
(*Easy* / *Moderate* / *Difficult*) — 734 from the original inventory, 206 added in a
second labeling pass, and 722 more added in a third pass (2026-08-28).

Two companion papers cover the work: a national review
([`report/geosite_ai_section_2026.pdf`](report/geosite_ai_section_2026.pdf)) and a
regional comparison
([`report/geosite_ai_section_2026_paper2_regional.pdf`](report/geosite_ai_section_2026_paper2_regional.pdf)).

## Results at a glance

| Model | *N* | Validation | Metric | Value |
|---|---:|---|---|---:|
| Geosite-location favorability | 1,667 | Spatial block CV | AUC | **0.956** |
| National Difficult vs. not (deployed: Gaussian Process + Infra features) | 1,662 | 10-fold group CV | Accuracy | **80.8%** |
| National Easy vs. not (deployed: tree ensemble + Infra features) | 1,662 | 500m LOGO-cluster CV | Accuracy | **73.5%** |
| Guelmim-Oued Noun + Laâyoune (Easy vs. not, merged) | 56 | 500m LOGO-cluster CV | Accuracy | 83.9% (+30.3pp vs. local baseline) |
| Tanger-Tétouan-Al Hoceima (Easy vs. not) | 312 | 500m LOGO-cluster CV | Accuracy | 79.2% (+27.9pp) |
| Fés-Meknés (Difficult vs. not) | 537 | 500m LOGO-cluster CV | Accuracy | 73.6% (+9.4pp) |

Every model uses the same core terrain/infrastructure feature stack (slope, ruggedness,
elevation, distance to highway, distance to settlement, land-cover friction --
extended nationally and regionally with geological-domain or tourism-infrastructure
features when that beats the plain baseline) and a 500m haversine-clustered
leave-one-group-out CV protocol throughout, specifically to avoid the near-duplicate-site
leakage that inflates naive random-split accuracy in spatial data. National numbers are
checked against a paired McNemar test; regional numbers are reported against each
region's own local majority-class baseline (see the regional paper for the full 20-row
region-by-region breakdown, including the five region/target combinations that do
*not* clear their local baseline, and four that tie it exactly).

| <img src="report/figures/map_national_preview.png" width="380" alt="National accessibility projection"> | <img src="report/figures/map_favorability_preview.png" width="380" alt="Geosite-location favorability"> |
|:---:|:---:|
| National accessibility projection (deployed GP+Infra / tree+Infra models) | Geosite-location favorability -- where terrain/geology resembles known geosite locations, not accessibility |
| <img src="report/figures/map_national_mosaic_preview.png" width="380" alt="Regional-mosaic accessibility map"> | <img src="report/figures/paper2_gap_chart_single_preview.png" width="380" alt="Regional accuracy vs. local majority baseline"> |
| Regional paper's national mosaic, assembled from per-region best models (Oriental shown as a disclosed national-model fallback, correct on 1 of its 4 labeled sites) | Single-region accuracy vs. each region's own local majority baseline, gap labeled |

Full-resolution figures: [`report/figures/`](report/figures/).

## Repository structure

The pipeline runs in three numbered stages, each in its own top-level folder,
meant to be run in order:

| Path | Contents |
|---|---|
| [`01_data_preparation/`](01_data_preparation/) | Catalog build: ingest the three labeling batches, extract terrain/road/settlement features, merge into the final catalog, build the master Excel, train the favorability model, run the model-family battery. Run `01`→`10` in order; `_catalog_helpers.py` is a shared library, not a step. `historical_one_shot_migrations/` documents three one-off column additions (Copernicus DEM, WorldCover LULC, settlement distance) already baked into `data/final/` — **not rerunnable**, their source inputs no longer exist; kept for provenance only. |
| [`02_modeling_and_analysis/`](02_modeling_and_analysis/) | Modeling, statistical testing, and calibration/conformal analysis. See its own [`README.md`](02_modeling_and_analysis/README.md) for run order — `01`/`02` are audit-trail records, `03`–`33` are the live pipeline (`30`–`33` added for the N=1,662 update: extended model-family comparison, the deployed models' per-class precision/recall, and the Oriental national-model fallback check). |
| [`03_report_generation/`](03_report_generation/) | Figure/map rendering scripts for both papers and the presentation. `region_infra_grid.py` streams the national OSM extract with `pyosmium` (a 2026-09-01 rewrite; the previous `pyrosm`-based version could not reliably complete a national-scale extraction). |
| [`report/`](report/) | The written deliverables — `geosite_ai_section_2026.tex`/`.pdf` (national review), `geosite_ai_section_2026_paper2_regional.tex`/`.pdf` (regional comparison), their figures (`figures/`), and the UM6P/GSMI logo (`um6p_figures/`). Both `.tex` files compile as-is from this folder. |
| [`presentation/`](presentation/) | French wrap-up slide deck (Beamer), `wrapup.tex`/`.pdf`. **Not yet updated for the N=1,662 catalog** — still reports the original 733-site batch's numbers; treat as a separate, older deliverable until refreshed. |
| [`data/`](data/) | `final/` — the labeled catalog and dataset used throughout (join key: `Locality_ID`), including `regional_label_sources/` (the three per-batch label files). `model_outputs/` — hyperparameter search results, prediction grids, favorability output. `boundaries/` — region GeoJSON boundaries used for map rendering. |
| [`models/final/`](models/final/) | The deployed favorability model (`geosite_location_pilot_model_v4.joblib`). |
| [`models/experimental/`](models/experimental/) *(gitignored)* | Earlier model iterations, kept locally for provenance, not tracked in git. |
| [`references/`](references/) | `databases/` — the source Excel/CSV geoheritage databases. `articles/` — cited papers (two large third-party copyrighted PDFs are gitignored, not redistributed). |
| [`results/`](results/) | Raw experiment outputs (JSON/CSV) that `02_modeling_and_analysis/` and `03_report_generation/` scripts read and write. |
| [`exploration/`](exploration/) | Side investigations not part of the reported pipeline: HDBSCAN clustering (`hdbscan/`) and the favorability-v3 notebook (`notebooks/`). |
| `archive/`, `livrable/` *(gitignored)* | `archive/superseded_scripts/` — ~25 dead-end/superseded scripts kept for history, see its `README.md`. `livrable/` — a standalone, self-contained handoff copy of the deliverable for the supervisors (same folder set as above, minus `archive/`/`exploration/`), refreshed alongside this README. Both kept locally, not pushed. |

## Reproduction

1. `01_data_preparation/` in numeric order (skip `historical_one_shot_migrations/`).
2. `02_modeling_and_analysis/` in numeric order (skip `01`/`02`).
3. `03_report_generation/*.py` to regenerate figures/maps, then compile either
   `.tex` in `report/` with `pdflatex` (run twice, for cross-references).

Each script locates the repo root via `os.path.dirname(os.path.abspath(__file__))`
plus a fixed number of `..` segments — keep the directory layout above intact.

## Methodology, in one paragraph

1,662 labeled geosites (734 from the original citation-traceable inventory, 206 added
in a second labeling pass, and 722 more added in a third pass on 2026-08-28), six core
terrain/infrastructure features common to every model: slope, ruggedness, elevation
(Copernicus GLO-30 DEM), distance to highway (OSM), distance to settlement (haversine
to reference settlement points), and land-cover friction (ESA WorldCover). The
baseline deployed model is a four-member soft-voted tree ensemble (Random Forest +
XGBoost + Gradient Boosting + LightGBM), class-balance-weighted; testing it against
three additional, structurally unrelated model families (logistic regression, *k*-NN,
and a Gaussian Process classifier) under the same 500m LOGO-cluster protocol found
that a Gaussian Process classifier, trained on the baseline features plus
tourism-infrastructure indicators, significantly beats the tree ensemble specifically
for the Difficult target ($p=0.0073$) — it is the model actually deployed and mapped
for that target nationally, while the tree ensemble (also with the infrastructure
features added) remains deployed for Easy. On the plain six-feature baseline, national
binary accuracy is 76.7% Difficult / 71.3% Easy; adding tourism-infrastructure
features raises both to 78.6% / 73.5%, each a statistically confirmed gain over the
baseline. A three-class model reaches 53.4% directly, or 58.7% with the
infrastructure features added — statistically indistinguishable from the
two-binary-classifier combination it was previously thought to edge out. Split-conformal
prediction reframes the deliverable from a single accuracy number to a calibrated
confidence statement: a confident call for roughly seven in ten Difficult predictions,
about one in two for Easy, an explicit field-verification flag for the rest. A
companion geosite-location favorability model (AUC 0.956, full 1,667-site catalog)
answers a related, distinct question — where currently uncatalogued geosites are
likely to be found. A full per-region breakdown, including which feature set and
region-merging strategy works best where, is the dedicated subject of the regional
companion paper, along with a robustness analysis tracing the national model's
recurring Difficult-class weakness to a specific, three-times-confirmed mechanism
(one region's terrain signature dominating the labeled training pool). Full detail
and honest discussion of limitations:
[`report/geosite_ai_section_2026.pdf`](report/geosite_ai_section_2026.pdf) and
[`report/geosite_ai_section_2026_paper2_regional.pdf`](report/geosite_ai_section_2026_paper2_regional.pdf).

## Acknowledgments

Work developed by [Mohamed El Gorrim](https://github.com/MedGm) under the supervision of Dr. Ismail Ben Amar , Mohamed El Ouali and Sanae El Harche at the Geology and Sustainable Mining Institute (GSMI), UM6P, as part of the UM6P AI/ML internship program.
