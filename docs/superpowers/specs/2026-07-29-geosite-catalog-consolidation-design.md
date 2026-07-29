# Geosite Catalog Consolidation — Design Spec

**Stage:** 1 of a multi-stage fresh-start rebuild of the Phase 1 national accessibility pipeline.
**Status:** Approved by user 2026-07-29 (round 3 revision — two-level Observation/Locality data model).

## Background

The prior Phase 1 rebuild (branch `worktree-phase1-foundations-fix`, unmerged) replaced the pipeline's circular ML label with real OSRM travel-time labels. The result flagged only **1 of 367** geosites as "Facile" (easy access) nationally. The user, with direct field knowledge of the Tanger-Tétouan-Al Hoceïma region, identified this as implausible: well-known, easily-accessible sites (Cap Malabata, Fahs-Anjra, Ras El Ma) were not showing up as Facile.

Investigation confirmed this is not purely a labeling-methodology problem — it is compounded by real, fixable data-quality bugs upstream:

- **Cap Malabata**: stored coordinate `lon=-0.0333` is corrupted (real value `-5.7133°W`, already documented in `livrable/phase2_regional_analytics/report/geosite_phase2_report_fr.tex:101` but never propagated back into any data file).
- **Ras Ma Spring / "Ras El Ma"**: exists with plausible coordinates in the *regional* TTAH catalog but is **absent from the national catalog** the pipeline actually trained on. A second, differently-coordinated "Source Ras El Ma" appears in the Phase 2 report's own sample table, unreconciled with the CSV.
- **Fahs-Anjra**: not a named catalog entry anywhere — it exists only as a coordinate point in the Phase 2 report's field-validation narrative (`geosite_phase2_report_fr.tex:289`), confirmed via real route-time verification (<750m / 11min walk from Gare de Melloussa) to be Facile, but this verified label was never written into any dataset.
- Two further report-documented, real-world-verified reclassification points exist unused in report prose: **Punta Cires** (Facile via piste driving distance) and **Oulmès/Khouribga plateau** (report itself flags this one as a likely overestimate — useful as a negative/cautionary example).
- The user separately flagged that **coordinate corruption is not limited to the sites above** — e.g. a Dakhla-Oued Eddahab geosite that appears to be placed near Laâyoune instead. This class of error is distinct from the border check: the point is still inside Morocco/Western Sahara, just in the wrong *region*, so a national border check alone would not catch it. This requires a systematic, nationwide regional-plausibility check (Process, below) — not a fix targeted at the one example spotted.

Full findings are recorded in `/home/medgm/um6p-intern/geosite_project1/collected_data/README.md`, alongside a consolidated inventory of every data file found across the project, assembled in `collected_data/main_checkout/` and `collected_data/worktree_task2_corrected/`.

### The `Data Classification_Geoheritage.xlsx` source is hierarchical, not flat

Direct inspection of this workbook (merged-cell ranges, real row content, and a hidden `Feuil2` sheet the author left behind as their own coordinate-conversion worked example) revealed a structure this spec must model explicitly:

- **`Data generale`** (primary sheet, 1133 rows, 392 merged ranges) has columns `Geological domain, Administrative region, Reference, Title, Evaluated geosites, Geosite's type, Coordinates (X_Easting, Y_Northing), Center coordinates (X_Easting, Y_Northing), Methodology, Comment`. `Domain`/`Region`/`Reference`/`Title` are merged down a whole group of rows — only the first row of each group carries a value; the rest are blank and must be forward-filled.
- Each row under `Evaluated geosites` is one **individual geological observation** (e.g. "Fault mirror", "Bioclastic limestone", "Current ripples"), with its own `Coordinates`. Multiple observation rows share one **`Center coordinates`** value — confirmed by the "Tinghir–Dades–Imilchil" group, where child observations (Lac Tislit, Todgha Gorges, ...) span **~70km** yet share a single declared center: the center is a locality/publication reference point, not a proxy for any individual observation's true position, and must never be substituted for one.
- **Coordinates are mixed-format and must be auto-detected per cell**, not assumed uniform: (a) projected Easting/Northing as plain numbers (e.g. `680098, 3380902`), (b) DMS with seconds (`32°35'34.12"N`), (c) DMS with **comma-decimal minutes**, a European convention (`31°06,87' N`), and (d) coordinate **intervals/ranges** (`Intervalle 31°20'0″N–32°40'0″N, ...`) which the source itself resolves to a midpoint — a fundamentally lower-precision case that should be flagged, not treated as equal-confidence to a real point.
- The projected coordinates' CRS was determined to be **UTM Zone 29N (EPSG:32629)** — confirmed two independent ways: converting `(680098, 3380902)` reproduces the already-known-correct "Eburnian-Proterozoic Unconformity" coordinate `(30.547, -7.122)` exactly, and the hidden `Feuil2` sheet states "Zone 29N" explicitly in its own remarks column.
- **`Feuil2` is not geosite data** — it's the source author's own legend/worked-example table for coordinate conversion, with 7 rows of `(raw string → correct decimal degrees)` pairs covering all the formats above, including the interval case. It should be used as a **validation fixture** for this stage's coordinate parser (parse each raw string, assert the result matches the stated decimal value within a small tolerance) — not ingested as a geosite record.
- **`Feuil1`** (784 rows, 220 merged ranges) has the same core data as `Data generale` but is missing the `Reference`/`Title`/`Methodology`/`Comment` columns and its row content matches `Data generale` row-for-row on the rows inspected. Working assumption: it's an earlier/simplified duplicate of `Data generale` and should be skipped in favor of the richer sheet — but this must be confirmed (not assumed) by comparing row counts and a sample of content during implementation, and reported either way.
- `Geosite's type` values are terse literature abbreviations (`SG`, `PT`, `PL`, `S`, `S & PL`, `SG&H`, ...) defined in a legend embedded in `Data generale` row 2. These must be carried through **verbatim** — no expansion or interpretation during this stage.

### Architectural decision: two-level data model (Observation + Locality/Geosite Group)

The user identified that immediately flattening every source into one geosite-per-row table is a one-way information loss: hierarchical sources like the Excel file above have real internal structure (many observations belonging to one locality), and collapsing that during ingestion — before cross-source matching even happens — would produce false duplicates or silently discard the hierarchy with no way to reconstruct it later.

This spec therefore defines **two distinct entities**, carried through the whole pipeline:

- **Observation**: one row from one source, as close to raw as normalization requires. Has its own coordinates, its own name/type, full provenance (source file, sheet, row). Never merged with another observation during loading — only later, deliberately, during locality construction.
- **Locality / Geosite Group**: the deduplicated entity that later stages (labeling, feature extraction, modeling) actually consume — "this is one physical place." Built by grouping observations via either **explicit hierarchy** the source itself states (the Excel `Center coordinates` grouping — strong evidence, not inferred) or **cross-source matching** (fuzzy name + spatial proximity, as in the original design) when the same real-world site appears as a standalone entry in multiple sources (e.g. "Lac Tislit" as its own row in both the Excel file and the national CSV catalog).

A locality can currently have exactly one observation (true for nearly every row in the national/TTAH/BMK CSV catalogs, where each catalog row already represents a distinct named site) or several (true for Excel groups like the Tinghir–Dades–Imilchil cluster). Nothing about this stage forces premature collapsing: a locality with only one observation today can gain more later without restructuring anything, and the full observation-level detail is never discarded.

The user has directed a fresh-start rebuild rather than continuing to patch the prior approach, with these hard constraints for the overall project:
- Use `.tif` GIS rasters and `.xlsx` reference data, cleaned of anything outside Moroccan borders.
- Road network: OSM national/provincial/autoroute classes only — no pistes. (Open question for a later stage: whether an official Moroccan government GIS source should supplement or replace OSM.)
- Final accessibility classes: Easy / Moderate / Difficult / Very Difficult.
- The user will contribute expert labels (starting with known-accessible TTAH sites) to validate the model, and more data than currently digitized may exist that the user can help surface.
- Model choice is open (including neural/deep learning) provided it is real-world efficient; cloud training capacity is available if needed.
- Work proceeds stage by stage, nothing implemented without explicit confirmation first.

This spec covers **only Stage 1**: producing a clean, trustworthy, hierarchy-preserving Observation table and the Localities derived from it. All later stages (road network sourcing, labeling protocol, feature engineering, modeling, validation, reporting) are out of scope here and will be brainstormed separately once Stage 1's output exists and has been reviewed.

## Goal

Produce (a) a complete, provenance-preserving table of every individual geosite **observation** from every known source, and (b) a deduplicated, borders-checked, region-consistency-checked **locality** table built from those observations — with every conflict, mismatch, ambiguous grouping, or judgment call surfaced for human review rather than silently auto-resolved. The locality table is what later stages build on; the observation table ensures nothing is ever permanently lost to premature flattening.

## Inputs

All read from the already-assembled `collected_data/` inventory (no re-fetching from the live project needed):

- `collected_data/main_checkout/livrable/phase1_national_accessibility/data/geosites_coordinates_clean.csv` (780 rows, national — each row is effectively a single-observation locality)
- `collected_data/main_checkout/livrable/phase2_regional_analytics/data/geosites_ttah_indexed.csv`
- `collected_data/main_checkout/livrable/phase2_regional_analytics/data/geosites_bmk_indexed.csv`
- `collected_data/main_checkout/references/Data Classification_Geoheritage.xlsx` — `Data generale` sheet (authoritative, hierarchical: observations + center-coordinate groupings, mixed UTM29N/DMS/interval coordinates); `Feuil1` (likely-duplicate, to be confirmed, skipped if confirmed redundant); `Feuil2` (not data — parser validation fixture only, 7 ground-truth conversion pairs)
- `collected_data/main_checkout/references/Morocco_Geosites_Graph_Data.xlsx` (checked for any coordinate data; primarily typology/domain breakdowns per earlier inspection)
- `collected_data/main_checkout/scratch/geosites_draa_tafilalet_indexed.csv`, `geosites_physical_features.csv`, `geosites_physical_features_general.csv`, `geosites_road_features.csv` (checked for any unique sites/coordinates not present elsewhere)
- Hand-entered ground-truth points from `geosite_phase2_report_fr.tex`: Fahs-Anjra/Melloussa `(35.7250, -5.6685)`, Punta Cires (already present in TTAH CSV, needs its report-documented Facile correction applied), Oulmès/Khouribga `(33.4720, -6.9462)` (kept as a flagged/cautionary example, not auto-applied as Facile)
- Morocco + Western Sahara boundary polygon (Natural Earth admin-0, same source already validated in the earlier registration-calibration work)
- Morocco's 12 administrative region boundary polygons (fetched live via OSM relations or Natural Earth admin-1 — new for this stage, needed for the regional-plausibility check)

## Process

### Phase A — Load observations (preserving hierarchy, never flattening)

1. **Normalize each source into the Observation schema**: `Observation_ID, Geosite_Name, Latitude_WGS84, Longitude_WGS84, Region, Geosite_Type, Geological_Domain, Locality_Center_Lat, Locality_Center_Lon, Locality_Group_Key, Source_File, Source_Row_Ref`. For the CSV catalogs (national, TTAH, BMK), each row becomes one observation with `Locality_Center_*` left null and `Locality_Group_Key` set to its own `Observation_ID` (it is its own trivial single-observation group). For the Excel source, `Locality_Group_Key` is set from the forward-filled `Reference`+`Center coordinates` combination shared by a run of rows, so the Tinghir–Dades–Imilchil-style groups are captured as explicit, source-stated groups — not inferred.
2. **Parse coordinates**, auto-detecting format per cell: plain numeric pairs as UTM Zone 29N Easting/Northing (converted via `pyproj`), DMS-with-seconds, DMS-with-comma-decimal-minutes, and interval/range strings (parsed to their midpoint but flagged with `Coordinate_Precision = "interval"` rather than `"point"`, so downstream consumers can treat them with appropriate skepticism). Anything matching none of these patterns is **not guessed at** — it goes to `dms_parse_failures.csv` for manual inspection. Validate the parser against `Feuil2`'s 7 ground-truth pairs before trusting it on the real data (all 7 must parse to within a small tolerance of the stated decimal value).
3. **Resolve the `Feuil1` vs `Data generale` question**: compare row counts and a content sample; if confirmed duplicate, parse only `Data generale` and report the comparison; if genuinely different, parse both and say so.
4. Load all other CSV/Excel sources per their existing flat schemas (each row = one observation = its own trivial locality group, as in Step 1).
5. **Emit the full Observation table**, unfiltered, unmerged — this is the traceability backbone for everything downstream.

### Phase B — Per-observation correctness checks

6. **Apply documented corrections.** Cap Malabata's longitude corrected to `-5.7133` per the Phase 2 report, with a `Correction_Applied` note recorded on that observation (not a silent overwrite).
7. **Border check.** Point-in-polygon test against the Morocco+Western Sahara boundary, applied per observation. Anything outside is removed and appended to `geosites_outliers_removed.csv` with its coordinates and source.
8. **Regional plausibility check — every observation, nationwide, not just the one flagged example.** Point-in-polygon test each observation's coordinates against all 12 Moroccan region boundaries; if the declared region and the geometrically-detected region disagree, the observation is excluded from locality construction and appended to `geosites_region_mismatches.csv` with both regions recorded.
9. **Fold in report-verified ground truth** as new observations (each its own single-observation locality group): Fahs-Anjra/Melloussa (Facile, cites the report and the Melloussa routing verification), Punta Cires' report-documented Facile correction applied to its existing observation (recording old and new category), Oulmès/Khouribga added with the report's own caution flagged, not silently labeled Facile.

### Phase C — Locality construction (deliberate, not automatic-by-default)

10. **Group observations into localities.** Two grouping mechanisms, applied in this order:
    - **Explicit source-stated grouping**: observations sharing a `Locality_Group_Key` from Step 1 (the Excel center-coordinate groups) are grouped into one locality by construction — this is not inferred, it is what the source document itself asserts.
    - **Cross-source fuzzy matching**: for observations that are each their own singleton group (the common case for the CSV catalogs and standalone Excel entries like "Lac Tislit"), compute fuzzy name similarity and spatial proximity across sources, same method as the original design. Compare **both** an observation's own coordinates and its locality's center coordinates (if any) against candidates, per the user's requirement that both signals inform matching. Agreeing matches merge into one locality, recording every contributing `Source_File`. Disagreeing or ambiguous matches (name similar, coordinates far apart — or vice versa) are **not merged** — all candidate observations involved go to `geosites_needs_review.csv` with the computed distance/similarity, for the user to resolve.
    - Explicit source-stated groups are never dissolved or reinterpreted by the fuzzy matcher — a locality that the Excel source already asserts (via shared center coordinates) is only ever *extended* by a cross-source match (e.g. if "Lac Tislit" the CSV entry matches the Excel's "Lac Tislit" observation), never re-split.
11. **Derive each locality's representative record**: name (prefer the CSV catalogs' name if the locality includes one, since those are typically the "official" name; otherwise the Excel observation's name), representative coordinates (the CSV catalog's coordinate if present — generally more precise/reviewed than a literature-derived point — otherwise the centroid of the locality's own observations, explicitly NOT the Excel `Locality_Center_*` field, which is a publication reference point, not a computed centroid), region, and the full list of member `Observation_ID`s.
12. **Emit sanity output.** The script prints, at minimum: observation counts per source before any grouping, sites removed as outliers (with examples), number of DMS/UTM parse failures and the Feuil2 validation result, number of region mismatches found nationwide (with examples), number of explicit-hierarchy locality groups found in the Excel source, number of cross-source fuzzy matches merged vs. sent to review, and final locality count vs. observation count. These are the "tests" for this stage, following the project's inline-assertion convention (e.g. `assert locality_count <= observation_count`, `assert no observation's coordinates fall outside the Morocco+WS polygon`, `assert Cap Malabata's longitude == -5.7133 in the final locality table`, `assert every Feuil2 test case parses within tolerance`).

## Outputs

All under a new directory, `geosite_v2_work/data/`:

- `geosites_observations.csv` — **every** individual observation from every source, post-correction and post-border/region-check, each tagged with its `Locality_Group_Key` (source-stated) and (once Phase C runs) its resolved `Locality_ID`. This is the full-detail, nothing-discarded table.
- `geosites_localities_master.csv` — one row per deduplicated locality: representative name/coordinates/region, member observation count, contributing sources. **This is what later stages (labeling, feature extraction, modeling) consume.**
- `geosites_needs_review.csv` — every unresolved cross-source matching conflict, side by side, for the user.
- `geosites_outliers_removed.csv` — audit trail of observations dropped for being outside Morocco.
- `geosites_region_mismatches.csv` — every observation whose declared region doesn't match its coordinate's actual location, nationwide.
- `dms_parse_failures.csv` — any coordinate string (DMS or otherwise) that couldn't be confidently parsed.

Code lives in `geosite_v2_work/code/01_consolidate_geosite_catalog.py`.

## Explicitly out of scope for this stage

- Road network sourcing decision (OSM vs. official government data) — separate stage.
- Terrain/distance feature extraction — separate stage, once the locality table is finalized.
- Labeling protocol and expert-label collection beyond the report-verified points already folded in — separate stage.
- Model architecture selection — separate stage.
- Expanding the Excel `Geosite_Type` abbreviations into full category names — kept verbatim this stage; expansion is a later enrichment step if ever needed.
- Resolving every entry in `geosites_needs_review.csv` or `geosites_region_mismatches.csv` automatically — both files are deliverables *to* the user, not something this stage resolves on its own.
- Collapsing an Excel-hierarchy locality's multiple observations (e.g. the 10+ features at the Tinghir–Dades–Imilchil cluster) into a single "geosite" for modeling purposes — that grouping already exists at the *observation* level via `Locality_Group_Key`/`Locality_ID`; deciding whether such a cluster should ultimately be scored as one accessibility point or many is a later-stage modeling decision, not this stage's to make.

## Open implementation details (not blocking approval, to be handled with judgment during implementation and reported clearly)

- Exact fuzzy-match similarity/distance thresholds for cross-source matching.
- Whether `Morocco_Geosites_Graph_Data.xlsx` and the `scratch/*.csv` files contribute any sites not already covered by the three main catalogs (to be determined by actually reading them during implementation).
- Precise DMS/UTM-parsing implementation, validated against `Feuil2`'s 7 ground-truth pairs before trusting it on real data.
- Whether `Feuil1` is confirmed redundant with `Data generale` (to be checked, not assumed, during implementation).
- Source and exact boundary vintage for the 12 regional polygons (OSM relation IDs vs. Natural Earth admin-1).
- Precise rule for choosing a locality's "representative coordinate" when a CSV catalog entry and an Excel observation both contribute to the same locality but disagree by a small amount (within cross-source matching tolerance, but not identical) — a reasonable default (prefer CSV catalog coordinate, documented per-row) is proposed in Process Step 11, open to revision once real examples are seen.

## Self-review

- **Placeholder scan:** none — every step has a concrete method and output.
- **Internal consistency:** the Observation/Locality distinction is threaded consistently through Inputs, Process, and Outputs — no step still assumes an immediate flat merge.
- **Scope check:** appropriately bounded to a single implementation plan — one script, six output files (five deliverables + the full observation table), no modeling or feature engineering.
- **Ambiguity check:** the genuine judgment calls (fuzzy-match thresholds, DMS/UTM parsing, `Feuil1` redundancy, regional-boundary source, representative-coordinate tie-breaking) are explicitly called out as "open implementation details," each with a proposed default or verification method rather than left hanging.
- **Addressed user feedback (round 2):** the regional-plausibility check is a systematic, nationwide sweep over every observation (Process Step 8), not a targeted fix for the single Dakhla/Laâyoune example.
- **Addressed user feedback (round 3):** replaced the original single-level "normalize then fuzzy-merge into one flat master catalog" design with the two-entity Observation/Locality model, specifically to avoid the information loss and false-duplicate risk the user identified. The Excel source's real hierarchical structure (verified by direct inspection: merged cells, the Tinghir–Dades–Imilchil 70km-spanning shared-center group, the hidden `Feuil2` conversion legend) is preserved end-to-end via `Locality_Group_Key`, never collapsed during loading — only deliberately grouped in the dedicated Phase C step, with source-stated groupings treated as strictly stronger evidence than inferred fuzzy matches and never overridden by them.
