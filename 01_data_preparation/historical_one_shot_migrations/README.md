# Historical one-shot migrations

These three scripts each mutated the production catalog exactly once, in
place, and were never meant to be rerun repeatedly — they're one-time source
swaps / column additions, not steps in a repeatable pipeline. Their effects
are permanently baked into `data/final/geosites_mcdm_national.csv` today:

| Script | What it did | Evidence it's baked in |
|---|---|---|
| `replace_terrain_with_copernicus_dem.py` (was `code/06`) | Replaced old scanned-map elevation/slope/ruggedness with real Copernicus GLO-30 DEM sampling | Current catalog's `Elevation_m`/`Slope_deg`/`Ruggedness` are DEM-consistent |
| `replace_lulc_with_worldcover.py` (was `code/07`) | Replaced the old LULC source with ESA WorldCover | Current catalog has the `LULC_Class_Name_WorldCover` column this script introduces |
| `compute_settlement_distance.py` (was `code/08`) | Added `Dist_to_Settlement_m` against 55 reference cities | Column present in the current catalog; the reference-city list is reused directly by `03_report_generation/make_maps.py` |

**None of these are rerunnable today as-is** — their own source input files
(`data/training/geosites_accessibility_dataset_v1.csv` and siblings) were
archived away in an earlier cleanup (`data/archive/superseded_training_data/`).
Kept here for provenance / methodology documentation only. If a supervisor
needs to reproduce the *current* catalog from scratch, they don't need to
rerun these — `data/final/geosites_mcdm_national.csv` already reflects their
effect; start from the live pipeline in the parent directory instead.
