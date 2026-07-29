# Geosite Catalog Consolidation — Design Spec

**Stage:** 1 of a multi-stage fresh-start rebuild of the Phase 1 national accessibility pipeline.
**Status:** Approved by user 2026-07-29.

## Background

The prior Phase 1 rebuild (branch `worktree-phase1-foundations-fix`, unmerged) replaced the pipeline's circular ML label with real OSRM travel-time labels. The result flagged only **1 of 367** geosites as "Facile" (easy access) nationally. The user, with direct field knowledge of the Tanger-Tétouan-Al Hoceïma region, identified this as implausible: well-known, easily-accessible sites (Cap Malabata, Fahs-Anjra, Ras El Ma) were not showing up as Facile.

Investigation confirmed this is not purely a labeling-methodology problem — it is compounded by real, fixable data-quality bugs upstream:

- **Cap Malabata**: stored coordinate `lon=-0.0333` is corrupted (real value `-5.7133°W`, already documented in `livrable/phase2_regional_analytics/report/geosite_phase2_report_fr.tex:101` but never propagated back into any data file).
- **Ras Ma Spring / "Ras El Ma"**: exists with plausible coordinates in the *regional* TTAH catalog but is **absent from the national catalog** the pipeline actually trained on. A second, differently-coordinated "Source Ras El Ma" appears in the Phase 2 report's own sample table, unreconciled with the CSV.
- **Fahs-Anjra**: not a named catalog entry anywhere — it exists only as a coordinate point in the Phase 2 report's field-validation narrative (`geosite_phase2_report_fr.tex:289`), confirmed via real route-time verification (<750m / 11min walk from Gare de Melloussa) to be Facile, but this verified label was never written into any dataset.
- Two further report-documented, real-world-verified reclassification points exist unused in report prose: **Punta Cires** (Facile via piste driving distance) and **Oulmès/Khouribga plateau** (report itself flags this one as a likely overestimate — useful as a negative/cautionary example).
- The user separately flagged that **coordinate corruption is not limited to the sites above** — e.g. a Dakhla-Oued Eddahab geosite that appears to be placed near Laâyoune instead. This class of error is distinct from the border check in this spec: the point is still inside Morocco/Western Sahara, just in the wrong *region*, so a national border check alone would not catch it. This requires a separate regional-plausibility check (added to the Process below).

Full findings are recorded in `/home/medgm/um6p-intern/geosite_project1/collected_data/README.md`, alongside a consolidated inventory of every data file found across the project (raw + Task-2-registration-corrected variants), assembled in `collected_data/main_checkout/` and `collected_data/worktree_task2_corrected/`.

The user has directed a fresh-start rebuild rather than continuing to patch the prior approach, with these hard constraints for the overall project:
- Use `.tif` GIS rasters and `.xlsx` reference data, cleaned of anything outside Moroccan borders.
- Road network: OSM national/provincial/autoroute classes only — no pistes. (Open question for a later stage: whether an official Moroccan government GIS source should supplement or replace OSM.)
- Final accessibility classes: Easy / Moderate / Difficult / Very Difficult.
- The user will contribute expert labels (starting with known-accessible TTAH sites) to validate the model, and more data than currently digitized may exist that the user can help surface.
- Model choice is open (including neural/deep learning) provided it is real-world efficient; cloud training capacity is available if needed.
- Work proceeds stage by stage, nothing implemented without explicit confirmation first.

This spec covers **only Stage 1**: producing one clean, trustworthy master geosite catalog. All later stages (road network sourcing, labeling protocol, feature engineering, modeling, validation, reporting) are out of scope here and will be brainstormed separately once Stage 1's output exists and has been reviewed.

## Goal

Produce a single, deduplicated, borders-checked, region-consistency-checked master geosite catalog — with every conflict, mismatch, or judgment call surfaced for human review rather than silently auto-resolved or auto-corrected — that later stages can build on with confidence.

## Inputs

All read from the already-assembled `collected_data/` inventory (no re-fetching from the live project needed):

- `collected_data/main_checkout/livrable/phase1_national_accessibility/data/geosites_coordinates_clean.csv` (780 rows, national)
- `collected_data/main_checkout/livrable/phase2_regional_analytics/data/geosites_ttah_indexed.csv`
- `collected_data/main_checkout/livrable/phase2_regional_analytics/data/geosites_bmk_indexed.csv`
- `collected_data/main_checkout/references/Data Classification_Geoheritage.xlsx` (`Data generale` / `Feuil1` sheets — has DMS-format coordinate strings, some malformed)
- `collected_data/main_checkout/references/Morocco_Geosites_Graph_Data.xlsx` (checked for any coordinate data; primarily typology/domain breakdowns per earlier inspection)
- `collected_data/main_checkout/scratch/geosites_draa_tafilalet_indexed.csv`, `geosites_physical_features.csv`, `geosites_physical_features_general.csv`, `geosites_road_features.csv` (checked for any unique sites/coordinates not present elsewhere)
- Hand-entered ground-truth points from `geosite_phase2_report_fr.tex`: Fahs-Anjra/Melloussa `(35.7250, -5.6685)`, Punta Cires (already present in TTAH CSV, needs its report-documented Facile correction applied), Oulmès/Khouribga `(33.4720, -6.9462)` (kept as a flagged/cautionary example, not auto-applied as Facile)
- Morocco + Western Sahara boundary polygon (Natural Earth admin-0, same source already validated in the earlier registration-calibration work — re-fetched live or reused from `collected_data/worktree_task2_corrected` if a cached copy exists there)
- Morocco's 12 administrative region boundary polygons (fetched live via OSM relations or Natural Earth admin-1 — new for this stage, needed for the regional-plausibility check below)

## Process

1. **Normalize.** Load each source into a common schema: `Geosite_Name, Latitude_WGS84, Longitude_WGS84, Region, Geosite_Type, Geological_Domain, Source_File`.
2. **Parse DMS coordinates** from the Excel reference file. Use a real DMS parser (handles `DD°MM'SS.SS"N/S/E/W` and common variants); any string that doesn't match a recognized DMS pattern (e.g. `513050N`) is **not guessed at** — it goes to a separate `dms_parse_failures.csv` for manual inspection, not silently dropped or silently included with a wrong value.
3. **Apply documented corrections.** Cap Malabata's longitude corrected to `-5.7133` per the Phase 2 report, with a `Correction_Applied` note recorded on that row (not a silent overwrite).
4. **Border check.** Point-in-polygon test against the Morocco+Western Sahara boundary. Anything outside is removed from the master catalog and appended to `geosites_outliers_removed.csv` with its coordinates and source, so nothing just vanishes unexplained.
5. **Regional plausibility check — applied to every site, nationwide, not just the one example already spotted.** For every site with a declared `Region`, point-in-polygon test its coordinates against all 12 Moroccan administrative region boundaries to find which region the coordinate *actually* falls inside. If the declared region and the geometrically-determined region disagree (e.g. declared "Dakhla-Oued Eddahab" but the point sits inside "Laâyoune-Sakia El Hamra"), the site is **excluded from the master catalog** and appended to `geosites_region_mismatches.csv` with both the declared and detected region, so it can't silently carry a wrong location into training later. This is a blanket sweep across the full dataset — no assumption is made that the Dakhla/Laâyoune case is the only instance.
6. **Cross-source matching.** For every pair of candidate sites (within-source and across-source), compute name similarity (fuzzy token match) and spatial distance. Pairs above a similarity/proximity threshold are treated as likely-the-same-site candidates:
   - If their coordinates agree within a small tolerance (e.g. a few hundred metres — exact figure to be tuned during implementation and shown in the script's output, not hidden), merge into a single row, keeping the union of available metadata and recording which sources agreed.
   - If they disagree beyond tolerance (e.g. the two "Ras El Ma"-type coordinate pairs), **do not merge or pick automatically** — both candidate rows go to `geosites_needs_review.csv` with source, name, coordinates, and the computed distance between them, for the user to resolve.
7. **Fold in report-verified ground truth.** Add Fahs-Anjra/Melloussa as a new named entry (Facile, with a provenance note citing the report and the Gare de Melloussa routing verification). Apply Punta Cires' report-documented Facile correction to its existing catalog row, recording both the old and new category. Add Oulmès/Khouribga as an entry flagged with the report's own caution about it being a likely overestimate — not blindly labeled Facile.
8. **Emit sanity output.** The script prints, at minimum: total sites per source before merge, sites removed as outliers (with a couple of examples), number of DMS parse failures, number of region mismatches found nationwide (with examples, not just the Dakhla/Laâyoune case), number of cross-source matches merged automatically vs. sent to review, and final master catalog size. These are the "tests" for this stage, following the project's existing convention of inline assertions rather than a pytest suite (e.g. `assert master_catalog_size > 367`, `assert no row's coordinates fall outside the Morocco+WS polygon`, `assert Cap Malabata's longitude == -5.7133 in the final catalog`, `assert every row's declared region matches its geometrically-detected region`).

## Outputs

All under a new directory, `livrable/phase1_v2_accessibility/data/`:

- `geosites_master_catalog.csv` — the clean, merged, in-borders, region-consistent catalog. This is what later stages consume.
- `geosites_needs_review.csv` — every unresolved cross-source conflict, side by side, for the user.
- `geosites_outliers_removed.csv` — audit trail of everything dropped for being outside Morocco.
- `geosites_region_mismatches.csv` — every site whose declared region doesn't match its coordinate's actual location, nationwide (declared region, geometrically-detected region, coordinates).
- `dms_parse_failures.csv` — any Excel DMS coordinate string that couldn't be confidently parsed.

Code lives in `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py`.

## Explicitly out of scope for this stage

- Road network sourcing decision (OSM vs. official government data) — separate stage.
- Terrain/distance feature extraction — separate stage, once the catalog is finalized.
- Labeling protocol and expert-label collection beyond the report-verified points already folded in — separate stage.
- Model architecture selection — separate stage.
- Resolving every entry in `geosites_needs_review.csv` or `geosites_region_mismatches.csv` automatically — both files are deliverables *to* the user, not something this stage resolves on its own.

## Open implementation details (not blocking approval, to be handled with judgment during implementation and reported clearly)

- Exact fuzzy-match similarity/distance thresholds for cross-source matching.
- Whether `Morocco_Geosites_Graph_Data.xlsx` and the `scratch/*.csv` files contribute any sites not already covered by the three main catalogs (to be determined by actually reading them during implementation).
- Precise DMS-parsing regex/library choice.
- Source and exact boundary vintage for the 12 regional polygons (OSM relation IDs vs. Natural Earth admin-1 — whichever gives cleaner, more complete coverage of all 12 regions including Western Sahara's, to be determined and reported during implementation).

## Self-review

- **Placeholder scan:** none — every step has a concrete method and output.
- **Internal consistency:** matches the design presented to and approved by the user; no contradictions between the process steps and the listed outputs.
- **Scope check:** appropriately bounded to a single implementation plan — one script, five output files, no modeling or feature engineering.
- **Ambiguity check:** the genuine judgment calls (fuzzy-match thresholds, DMS parsing, regional-boundary source) are explicitly called out as "open implementation details" rather than hidden, and all are the kind of thing the implementer should report on (thresholds used, parse failures found, mismatches found) rather than silently decide.
- **Addressed user feedback (round 2):** the regional-plausibility check was added as a systematic, nationwide sweep over every site (Step 5) — not a targeted fix for the single Dakhla/Laâyoune example the user happened to notice. The check is symmetric with the border check: mismatched sites are excluded from the master catalog and routed to a dedicated review file, never silently corrected or silently kept with a wrong region.
