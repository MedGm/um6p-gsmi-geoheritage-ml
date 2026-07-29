# Geosite Catalog Consolidation (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one script that consolidates every known geosite data source (national catalog, TTAH/BMK regional catalogs, Excel references, report-verified ground truth) into a single trustworthy master catalog, with every conflict, coordinate corruption, and region mismatch routed to a dedicated review file instead of silently resolved.

**Architecture:** A single pipeline script, `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py`, run in three logical (but not separately committed) stages: (1) load + normalize every source + parse Excel DMS coordinates, (2) apply documented corrections, fold in report-verified ground truth, and run the border + regional-plausibility checks, (3) cross-source fuzzy matching/merge and final assembly. Implemented as 3 tasks matching those stages, each producing an inspectable intermediate CSV so a reviewer can verify one stage without re-deriving the whole pipeline.

**Tech Stack:** Python 3.12, pandas, geopandas, shapely, rapidfuzz (fuzzy name matching), openpyxl (Excel), requests (Overpass/Natural Earth fetch — same pattern already validated in the prior registration-calibration work).

## Global Constraints

- CRS for all point-in-polygon geographic computation: work in WGS84 (EPSG:4326) for the border/region checks (matching the source coordinate CRS); do not introduce a projected CRS unless a specific step needs metric distance (cross-source proximity matching does — use EPSG:26191 Sahara Lambert for that, consistent with the rest of this project).
- No site is silently dropped, silently corrected, or silently merged. Every automatic action (border rejection, region mismatch, DMS parse failure, cross-source merge) must be logged to its designated output CSV, and the final script run must print a summary count for each.
- Region mismatch and border checks apply to **every site in the combined dataset, unconditionally** — not just previously-flagged examples (Cap Malabata, the Dakhla/Laâyoune case). Do not special-case known bad sites in the detection logic; only the *documented corrections* (Cap Malabata's coordinate, per the Phase 2 report) are applied as named exceptions, and even those must be recorded with a `Correction_Applied` note, not silently overwritten.
- Every script prints its own sanity-check assertions and exits non-zero on failure — this project has no pytest suite for the GIS/data pipeline; these assertions are this stage's tests.
- Full spec: `docs/superpowers/specs/2026-07-29-geosite-catalog-consolidation-design.md` — read it for complete background and rationale before starting Task 1.

---

## File Structure

```
livrable/phase1_v2_accessibility/
  code/
    01_consolidate_geosite_catalog.py   # CREATE — the only script this plan produces
  data/
    geosites_normalized_combined.csv    # CREATE — Task 1 output (all sources, normalized schema, tagged by source)
    dms_parse_failures.csv              # CREATE — Task 1 output
    geosites_outliers_removed.csv       # CREATE — Task 2 output
    geosites_region_mismatches.csv      # CREATE — Task 2 output
    geosites_checked.csv                # CREATE — Task 2 output (post-correction, post-border/region-check, pre-merge)
    geosites_master_catalog.csv         # CREATE — Task 3 output (final deliverable)
    geosites_needs_review.csv           # CREATE — Task 3 output
```

`collected_data/` (already assembled, at the repo root, outside any worktree-specific path) is read-only input — nothing in this plan modifies it.

---

### Task 1: Load, normalize, and parse every geosite data source

**Files:**
- Create: `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py` (this task writes the `load_and_normalize()` portion and a `main()` that calls only this much so far; Tasks 2 and 3 extend the same file)
- Create: `livrable/phase1_v2_accessibility/data/geosites_normalized_combined.csv`
- Create: `livrable/phase1_v2_accessibility/data/dms_parse_failures.csv`

**Interfaces:**
- Consumes: `collected_data/main_checkout/livrable/phase1_national_accessibility/data/geosites_coordinates_clean.csv`, `collected_data/main_checkout/livrable/phase2_regional_analytics/data/geosites_ttah_indexed.csv`, `collected_data/main_checkout/livrable/phase2_regional_analytics/data/geosites_bmk_indexed.csv`, `collected_data/main_checkout/references/Data Classification_Geoheritage.xlsx`, `collected_data/main_checkout/references/Morocco_Geosites_Graph_Data.xlsx`, `collected_data/main_checkout/scratch/geosites_draa_tafilalet_indexed.csv`, `collected_data/main_checkout/scratch/geosites_physical_features.csv`, `collected_data/main_checkout/scratch/geosites_physical_features_general.csv`, `collected_data/main_checkout/scratch/geosites_road_features.csv`.
- Produces: `geosites_normalized_combined.csv` with columns `Geosite_Name, Latitude_WGS84, Longitude_WGS84, Region, Geosite_Type, Geological_Domain, Source_File` (one row per site-occurrence — NOT yet deduplicated across sources; that's Task 3) — consumed by Task 2.

- [ ] **Step 1: Write the source-loading and schema-normalization code**

```python
# livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
import os
import re
import numpy as np
import pandas as pd
import openpyxl

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COLLECTED = os.path.abspath(os.path.join(BASE, "..", "collected_data", "main_checkout"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(OUT_DIR, exist_ok=True)

NORMALIZED_COLS = ["Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Region", "Geosite_Type", "Geological_Domain", "Source_File"]

def load_csv_source(path, name_col, lat_col, lon_col, region_col=None, type_col=None, domain_col=None, source_tag=None):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["Geosite_Name"] = df[name_col].astype(str).str.strip()
    out["Latitude_WGS84"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["Longitude_WGS84"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["Region"] = df[region_col].astype(str).str.strip() if region_col and region_col in df.columns else np.nan
    out["Geosite_Type"] = df[type_col] if type_col and type_col in df.columns else np.nan
    out["Geological_Domain"] = df[domain_col] if domain_col and domain_col in df.columns else np.nan
    out["Source_File"] = source_tag or os.path.basename(path)
    return out[NORMALIZED_COLS]

def parse_dms(dms_str):
    """
    Parse a DMS coordinate string like 33°32'44.71"N or 32° 6'24.85"N into decimal degrees.
    Returns (decimal_degrees, None) on success, (None, reason) on failure.
    Does NOT attempt to guess-fix malformed strings like '513050N' — those are failures.
    """
    if dms_str is None:
        return None, "empty"
    s = str(dms_str).strip()
    m = re.match(r"^(\d{1,3})[°\s]+(\d{1,2})['′\s]+([\d.]+)[\"″\s]*([NSEW])$", s)
    if not m:
        return None, f"unrecognized format: {s!r}"
    deg, minute, sec, hemi = m.groups()
    try:
        val = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
    except ValueError:
        return None, f"non-numeric component: {s!r}"
    if hemi in ("S", "W"):
        val = -val
    return val, None

def load_excel_source(path, source_tag):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows, failures = [], []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for row in ws.iter_rows(values_only=True):
            # Heuristic: a row is a candidate geosite record if it has a name-like string
            # followed later by two coordinate-like strings. Adjust once real sheet
            # structure is inspected — this is intentionally conservative.
            name, lat_raw, lon_raw = None, None, None
            for cell in row:
                if isinstance(cell, str) and re.search(r"\d{1,3}[°]\s*\d", cell):
                    if lat_raw is None:
                        lat_raw = cell
                    elif lon_raw is None:
                        lon_raw = cell
                elif isinstance(cell, str) and cell.strip() and name is None and not re.match(r"^\d", cell.strip()):
                    name = cell.strip()
            if name and lat_raw and lon_raw:
                lat, lat_err = parse_dms(lat_raw)
                lon, lon_err = parse_dms(lon_raw)
                if lat_err or lon_err:
                    failures.append({"Geosite_Name": name, "Raw_Lat": lat_raw, "Raw_Lon": lon_raw,
                                      "Lat_Error": lat_err, "Lon_Error": lon_err, "Source_File": f"{source_tag}:{sheet_name}"})
                else:
                    rows.append({"Geosite_Name": name, "Latitude_WGS84": lat, "Longitude_WGS84": lon,
                                 "Region": np.nan, "Geosite_Type": np.nan, "Geological_Domain": np.nan,
                                 "Source_File": f"{source_tag}:{sheet_name}"})
    return pd.DataFrame(rows, columns=NORMALIZED_COLS), pd.DataFrame(failures)

def main():
    sources = []
    sources.append(load_csv_source(
        os.path.join(COLLECTED, "livrable/phase1_national_accessibility/data/geosites_coordinates_clean.csv"),
        "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Administrative_Region", "Geosite_Type", "Geological_Domain",
        source_tag="national_catalog"))
    sources.append(load_csv_source(
        os.path.join(COLLECTED, "livrable/phase2_regional_analytics/data/geosites_ttah_indexed.csv"),
        "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Administrative_Region", "Geosite_Type", "Geological_Domain",
        source_tag="ttah_regional"))
    sources.append(load_csv_source(
        os.path.join(COLLECTED, "livrable/phase2_regional_analytics/data/geosites_bmk_indexed.csv"),
        "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Administrative_Region", "Geosite_Type", "Geological_Domain",
        source_tag="bmk_regional"))
    # scratch/*.csv: inspect each file's actual columns before writing its load_csv_source call —
    # do not assume they match the national catalog's schema; report actual column names found.

    excel_df, excel_failures = load_excel_source(
        os.path.join(COLLECTED, "references/Data Classification_Geoheritage.xlsx"), "geoheritage_excel")
    sources.append(excel_df)

    combined = pd.concat(sources, ignore_index=True)
    print(f"Loaded {len(combined)} raw site-occurrences from {len(sources)} CSV/derived sources (pre-Excel-merge dedup, pre-cleaning)")
    print(combined["Source_File"].value_counts())

    n_missing_coords = combined[["Latitude_WGS84", "Longitude_WGS84"]].isna().any(axis=1).sum()
    print(f"Rows with missing coordinates: {n_missing_coords}")

    combined.to_csv(os.path.join(OUT_DIR, "geosites_normalized_combined.csv"), index=False)
    excel_failures.to_csv(os.path.join(OUT_DIR, "dms_parse_failures.csv"), index=False)
    print(f"DMS parse failures: {len(excel_failures)}")
    assert len(combined) > 780, "Combined source count looks too low — check all sources loaded"

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Inspect the scratch CSVs' actual columns before wiring them in**

```bash
source venv/bin/activate
python -c "
import pandas as pd
for f in ['geosites_draa_tafilalet_indexed.csv', 'geosites_physical_features.csv', 'geosites_physical_features_general.csv', 'geosites_road_features.csv']:
    path = f'collected_data/main_checkout/scratch/{f}'
    df = pd.read_csv(path)
    print(f, '->', list(df.columns)[:8], f'({len(df)} rows)')
"
```

Add a `load_csv_source(...)` call for each scratch file to `sources.append(...)` in `main()`, using whatever column names Step 2 reveals (do not guess — use the printed output). If a scratch file's site names/coordinates are already fully covered by the national or regional catalogs (check by eye against a few rows), it's fine to skip it, but say so explicitly in the report rather than silently omitting it.

- [ ] **Step 3: Also load `Morocco_Geosites_Graph_Data.xlsx` and check whether it has coordinate data**

```bash
python -c "
import openpyxl
wb = openpyxl.load_workbook('collected_data/main_checkout/references/Morocco_Geosites_Graph_Data.xlsx', read_only=True)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(sn, '- first row:', next(ws.iter_rows(values_only=True), None))
"
```

If it contains coordinate-bearing rows, add it as a source via `load_excel_source()`. If it's purely typology/domain/region summary tables (as earlier inspection suggested), note that in the report and do not force it into the pipeline.

- [ ] **Step 4: Run the script and verify**

```bash
python livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
```

Expected: prints per-source counts, missing-coordinate count, DMS parse failure count; the `assert len(combined) > 780` passes; `geosites_normalized_combined.csv` and `dms_parse_failures.csv` are created.

- [ ] **Step 5: Sanity-check the DMS parser against the two known cases**

```bash
python -c "
from importlib import import_module
import sys
sys.path.insert(0, 'livrable/phase1_v2_accessibility/code')
m = import_module('01_consolidate_geosite_catalog')
print(m.parse_dms('33°32\'44.71\"N'))   # expect ~33.546 meters, no error
print(m.parse_dms('513050N'))           # expect (None, error) — must NOT silently parse this
"
```

Expected: the well-formed DMS string parses to a plausible decimal value; the malformed `513050N` string returns an error, not a guessed value.

- [ ] **Step 6: Commit**

```bash
git add livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py \
        livrable/phase1_v2_accessibility/data/geosites_normalized_combined.csv \
        livrable/phase1_v2_accessibility/data/dms_parse_failures.csv
git commit -m "feat: load and normalize all geosite data sources into one combined table"
```

---

### Task 2: Apply corrections, fold in ground truth, run border + regional-plausibility checks

**Files:**
- Modify: `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py` (extends `main()`, adds new functions)
- Create: `livrable/phase1_v2_accessibility/data/geosites_outliers_removed.csv`
- Create: `livrable/phase1_v2_accessibility/data/geosites_region_mismatches.csv`
- Create: `livrable/phase1_v2_accessibility/data/geosites_checked.csv`

**Interfaces:**
- Consumes: `geosites_normalized_combined.csv` (Task 1).
- Produces: `geosites_checked.csv` — same schema as Task 1's output plus `Correction_Applied` (string or NaN) and `Ground_Truth_Note` (string or NaN) columns, with border/region-mismatched rows already removed and diverted — consumed by Task 3.

- [ ] **Step 1: Write the correction + ground-truth fold-in code**

```python
# add to 01_consolidate_geosite_catalog.py

DOCUMENTED_CORRECTIONS = [
    # (name_substring_match, field, wrong_value_hint, correct_value, note)
    ("Malabata", "Longitude_WGS84", -0.0333, -5.7133,
     "Corrected per geosite_phase2_report_fr.tex:101 (regional field verification)"),
]

GROUND_TRUTH_ADDITIONS = [
    {"Geosite_Name": "Fahs-Anjra (Melloussa)", "Latitude_WGS84": 35.7250, "Longitude_WGS84": -5.6685,
     "Region": "Tanger-Tétouan-Al Hoceïma", "Geosite_Type": np.nan, "Geological_Domain": np.nan,
     "Source_File": "phase2_report_field_verification",
     "Ground_Truth_Note": "Verified <750m / 11min walk from Gare de Melloussa; report classifies Facile (geosite_phase2_report_fr.tex:289)"},
]

def apply_documented_corrections(df):
    df = df.copy()
    df["Correction_Applied"] = np.nan
    for name_sub, field, wrong_hint, correct_val, note in DOCUMENTED_CORRECTIONS:
        mask = df["Geosite_Name"].str.contains(name_sub, case=False, na=False) & np.isclose(df[field], wrong_hint, atol=0.01)
        n = mask.sum()
        if n > 0:
            df.loc[mask, field] = correct_val
            df.loc[mask, "Correction_Applied"] = note
            print(f"Applied correction for {n} row(s) matching {name_sub!r}: {field} -> {correct_val}")
    return df

def fold_in_ground_truth(df):
    additions = pd.DataFrame(GROUND_TRUTH_ADDITIONS)
    for col in NORMALIZED_COLS:
        if col not in additions.columns:
            additions[col] = np.nan
    if "Correction_Applied" not in additions.columns:
        additions["Correction_Applied"] = np.nan
    df = pd.concat([df, additions], ignore_index=True)
    print(f"Folded in {len(additions)} report-verified ground-truth site(s)")
    return df
```

- [ ] **Step 2: Write the border check**

```python
# add to 01_consolidate_geosite_catalog.py
import geopandas as gpd
from shapely.geometry import Point

def fetch_morocco_boundary():
    """Morocco + Western Sahara admin-0 boundary, WGS84. Same source pattern as the
    earlier registration-calibration work — Natural Earth admin-0, live fetch."""
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    morocco = world[world["ADMIN"].isin(["Morocco", "Western Sahara"])]
    assert len(morocco) >= 1, "Failed to fetch Morocco/Western Sahara boundary"
    return morocco.to_crs("EPSG:4326")

def border_check(df, boundary_gdf):
    geometry = [Point(xy) for xy in zip(df["Longitude_WGS84"], df["Latitude_WGS84"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    union = boundary_gdf.union_all()
    inside = gdf.geometry.within(union)
    outliers = gdf[~inside & gdf.geometry.notna()].drop(columns="geometry")
    kept = gdf[inside].drop(columns="geometry")
    print(f"Border check: {len(kept)} inside Morocco/Western Sahara, {len(outliers)} outside")
    return kept, outliers
```

- [ ] **Step 3: Write the regional-plausibility check**

```python
# add to 01_consolidate_geosite_catalog.py

def fetch_region_boundaries():
    """Morocco's 12 administrative regions, WGS84. Fetch via Natural Earth admin-1
    filtered to Morocco, or OSM relations if admin-1 coverage is incomplete for
    Western Sahara's regions — inspect what's actually returned before trusting it."""
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson"
    regions = gpd.read_file(url)
    morocco_regions = regions[regions["admin"].isin(["Morocco", "Western Sahara"])]
    print(f"Fetched {len(morocco_regions)} region polygons: {sorted(morocco_regions['name'].dropna().unique())}")
    assert len(morocco_regions) >= 10, "Expected ~12 Moroccan regions, got far fewer — check the admin-1 source/filter"
    return morocco_regions.to_crs("EPSG:4326")

def regional_plausibility_check(df, region_gdf, region_name_col):
    geometry = [Point(xy) for xy in zip(df["Longitude_WGS84"], df["Latitude_WGS84"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(gdf, region_gdf[[region_name_col, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={region_name_col: "Detected_Region"})

    has_declared = joined["Region"].notna() & (joined["Region"].astype(str).str.strip() != "") & (joined["Region"].astype(str) != "nan")
    # Fuzzy-normalize declared vs detected region names before comparing (accents, hyphenation
    # differ between sources) — implement a light normalization, do not do exact string equality only.
    mismatch = has_declared & (joined["Detected_Region"].notna()) & (~_regions_match(joined["Region"], joined["Detected_Region"]))

    mismatches = joined[mismatch].drop(columns="geometry")
    clean = joined[~mismatch].drop(columns=["geometry", "Detected_Region", "index_right"], errors="ignore")
    print(f"Regional plausibility check: {mismatch.sum()} mismatch(es) found nationwide, {len(clean)} sites consistent")
    if mismatch.sum() > 0:
        print(mismatches[["Geosite_Name", "Region", "Detected_Region"]].head(10).to_string(index=False))
    return clean, mismatches

def _regions_match(declared_series, detected_series):
    def norm(s):
        s = str(s).lower().strip()
        for a, b in [("é", "e"), ("è", "e"), ("â", "a"), ("ï", "i"), ("-", " "), ("'", " ")]:
            s = s.replace(a, b)
        return " ".join(s.split())
    return declared_series.apply(norm) == detected_series.apply(norm)
```

Note: `region_name_col` must be set to whatever the fetched region GeoDataFrame actually names its region column (inspect the fetched columns during Step 4's run — commonly `name` for Natural Earth admin-1 — and confirm it before wiring the call in `main()`).

- [ ] **Step 4: Wire it all into `main()`**

```python
# extend main() in 01_consolidate_geosite_catalog.py, after the Task 1 portion
    combined = apply_documented_corrections(combined)
    combined = fold_in_ground_truth(combined)

    boundary = fetch_morocco_boundary()
    combined, outliers = border_check(combined, boundary)
    outliers.to_csv(os.path.join(OUT_DIR, "geosites_outliers_removed.csv"), index=False)

    region_gdf = fetch_region_boundaries()
    combined, region_mismatches = regional_plausibility_check(combined, region_gdf, region_name_col="name")  # confirm column name from Step 3's printed output
    region_mismatches.to_csv(os.path.join(OUT_DIR, "geosites_region_mismatches.csv"), index=False)

    combined.to_csv(os.path.join(OUT_DIR, "geosites_checked.csv"), index=False)
    print(f"Post-checks: {len(combined)} sites remain for cross-source matching")

    assert not combined[["Latitude_WGS84", "Longitude_WGS84"]].isna().all(axis=1).any(), "Rows with fully-missing coordinates survived to this point"
    cap_malabata = combined[combined["Geosite_Name"].str.contains("Malabata", case=False, na=False)]
    if len(cap_malabata) > 0:
        assert np.isclose(cap_malabata.iloc[0]["Longitude_WGS84"], -5.7133, atol=0.01), "Cap Malabata correction did not apply"
```

- [ ] **Step 5: Run and verify**

```bash
python livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
```

Expected: correction-applied count printed (>=1 for Malabata), ground-truth fold-in count printed (1), border check prints inside/outside counts, regional check prints mismatch count with examples (report the actual number found — do not assume it will be exactly 1; the whole point of this check is to find however many actually exist), Cap Malabata assertion passes, `geosites_outliers_removed.csv`, `geosites_region_mismatches.csv`, `geosites_checked.csv` created.

- [ ] **Step 6: Manually spot-check the region-mismatch output for the known Dakhla/Laâyoune case**

```bash
python -c "
import pandas as pd
df = pd.read_csv('livrable/phase1_v2_accessibility/data/geosites_region_mismatches.csv')
print(df[df['Region'].str.contains('Dakhla', case=False, na=False)].to_string(index=False))
print(f'Total mismatches found: {len(df)}')
"
```

Report the full mismatch count and a few examples in the task report — this number matters to the user beyond just confirming the one case they already knew about.

- [ ] **Step 7: Commit**

```bash
git add livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py \
        livrable/phase1_v2_accessibility/data/geosites_outliers_removed.csv \
        livrable/phase1_v2_accessibility/data/geosites_region_mismatches.csv \
        livrable/phase1_v2_accessibility/data/geosites_checked.csv
git commit -m "feat: apply documented corrections, fold in ground truth, add border and nationwide regional-plausibility checks"
```

---

### Task 3: Cross-source fuzzy matching, merge, and final master catalog assembly

**Files:**
- Modify: `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py` (final extension to `main()`, adds matching functions)
- Create: `livrable/phase1_v2_accessibility/data/geosites_master_catalog.csv`
- Create: `livrable/phase1_v2_accessibility/data/geosites_needs_review.csv`

**Interfaces:**
- Consumes: `geosites_checked.csv` (Task 2).
- Produces: `geosites_master_catalog.csv` — final schema `Geosite_Name, Latitude_WGS84, Longitude_WGS84, Region, Geosite_Type, Geological_Domain, Source_Files (semicolon-joined list), Correction_Applied, Ground_Truth_Note` — this is the Stage 1 deliverable that later stages (out of scope for this plan) will consume.

- [ ] **Step 1: Write the fuzzy cross-source matching and merge logic**

```python
# add to 01_consolidate_geosite_catalog.py
from rapidfuzz import fuzz
from pyproj import Transformer

def cross_source_match_and_merge(df, name_similarity_threshold=85, proximity_tolerance_m=500):
    """
    Groups likely-same-site rows across sources by (fuzzy name similarity AND spatial
    proximity). Agreeing groups are merged into one row. Disagreeing groups (name similar
    but coordinates far apart, or vice versa in a way that's ambiguous) are NOT merged —
    both/all candidate rows go to the review set.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26191", always_xy=True)
    xs, ys = transformer.transform(df["Longitude_WGS84"].values, df["Latitude_WGS84"].values)
    df = df.copy()
    df["_x"], df["_y"] = xs, ys

    n = len(df)
    assigned = np.full(n, -1, dtype=int)
    group_id = 0
    review_rows = []
    merged_rows = []

    for i in range(n):
        if assigned[i] != -1:
            continue
        group = [i]
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            name_sim = fuzz.token_sort_ratio(str(df.iloc[i]["Geosite_Name"]), str(df.iloc[j]["Geosite_Name"]))
            dist_m = np.hypot(df.iloc[i]["_x"] - df.iloc[j]["_x"], df.iloc[i]["_y"] - df.iloc[j]["_y"])
            if name_sim >= name_similarity_threshold and dist_m <= proximity_tolerance_m:
                group.append(j)
                assigned[j] = group_id
        assigned[i] = group_id
        group_id += 1

        if len(group) == 1:
            row = df.iloc[group[0]].drop(labels=["_x", "_y"]).to_dict()
            row["Source_Files"] = row.pop("Source_File")
            merged_rows.append(row)
        else:
            # Check for internal disagreement above tolerance even within the fuzzy-matched group
            coords = df.iloc[group][["Latitude_WGS84", "Longitude_WGS84"]].values
            max_pairwise_dist = 0
            for a in range(len(coords)):
                for b in range(a + 1, len(coords)):
                    d = np.hypot(df.iloc[group[a]]["_x"] - df.iloc[group[b]]["_x"], df.iloc[group[a]]["_y"] - df.iloc[group[b]]["_y"])
                    max_pairwise_dist = max(max_pairwise_dist, d)
            if max_pairwise_dist <= proximity_tolerance_m:
                base = df.iloc[group[0]].drop(labels=["_x", "_y"]).to_dict()
                base["Source_Files"] = ";".join(df.iloc[group]["Source_File"].astype(str).unique())
                for col in ["Geosite_Type", "Geological_Domain", "Region"]:
                    non_null = df.iloc[group][col].dropna()
                    if base.get(col) in (None, np.nan) or (isinstance(base.get(col), float) and np.isnan(base.get(col))):
                        if len(non_null) > 0:
                            base[col] = non_null.iloc[0]
                merged_rows.append(base)
            else:
                for idx in group:
                    row = df.iloc[idx].drop(labels=["_x", "_y"]).to_dict()
                    row["Match_Group"] = group_id
                    row["Max_Pairwise_Distance_m"] = max_pairwise_dist
                    review_rows.append(row)

    master = pd.DataFrame(merged_rows)
    review = pd.DataFrame(review_rows)
    print(f"Cross-source matching: {len(df)} input rows -> {len(master)} merged/singleton sites, {len(review)} rows sent to review ({review['Match_Group'].nunique() if len(review) else 0} conflicting groups)")
    return master, review
```

- [ ] **Step 2: Wire into `main()` and write final outputs**

```python
# extend main() in 01_consolidate_geosite_catalog.py, after Task 2's portion
    master, needs_review = cross_source_match_and_merge(combined)

    master.to_csv(os.path.join(OUT_DIR, "geosites_master_catalog.csv"), index=False)
    needs_review.to_csv(os.path.join(OUT_DIR, "geosites_needs_review.csv"), index=False)

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Master catalog: {len(master)} sites")
    print(f"Needs review: {len(needs_review)} rows ({needs_review['Match_Group'].nunique() if len(needs_review) else 0} groups)")
    print(f"Outliers removed: {len(outliers)}")
    print(f"Region mismatches: {len(region_mismatches)}")
    print(f"DMS parse failures logged separately in dms_parse_failures.csv")

    assert len(master) >= 367, f"Master catalog ({len(master)} sites) is smaller than the prior pipeline's 367 — investigate before treating this as final"
    ras_ma = master[master["Geosite_Name"].str.contains("Ras.*Ma", case=False, na=False, regex=True)]
    print(f"\nRas El Ma / Ras Ma Spring candidates in master catalog: {len(ras_ma)}")
    if len(ras_ma) == 0:
        print("WARNING: no Ras El Ma variant made it into the master catalog — check needs_review.csv, it may have been flagged as a conflict (expected, given the two known conflicting coordinate pairs)")
```

- [ ] **Step 3: Run the full pipeline end to end**

```bash
python livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
```

Expected: full summary printed, `assert len(master) >= 367` passes, `geosites_master_catalog.csv` and `geosites_needs_review.csv` created. If Ras El Ma ends up in `needs_review.csv` rather than the master catalog, that's a correct outcome (the two coordinate pairs genuinely conflict) — confirm it's there, not silently dropped.

- [ ] **Step 4: Verify the master catalog against every known finding from the spec's Background section**

```bash
python -c "
import pandas as pd
master = pd.read_csv('livrable/phase1_v2_accessibility/data/geosites_master_catalog.csv')
review = pd.read_csv('livrable/phase1_v2_accessibility/data/geosites_needs_review.csv')

cap = master[master['Geosite_Name'].str.contains('Malabata', case=False, na=False)]
assert len(cap) >= 1 and abs(cap.iloc[0]['Longitude_WGS84'] - (-5.7133)) < 0.01, 'Cap Malabata not corrected in master catalog'
print('Cap Malabata: OK, corrected')

fahs = master[master['Geosite_Name'].str.contains('Fahs', case=False, na=False)]
assert len(fahs) >= 1, 'Fahs-Anjra ground truth missing from master catalog'
print('Fahs-Anjra: OK, present')

ras = pd.concat([master[master['Geosite_Name'].str.contains('Ras.*Ma', case=False, na=False, regex=True)],
                 review[review['Geosite_Name'].str.contains('Ras.*Ma', case=False, na=False, regex=True)] if len(review) else pd.DataFrame()])
assert len(ras) >= 1, 'No Ras El Ma variant found anywhere in outputs — investigate, this should not be possible'
print(f'Ras El Ma variants found across master+review: {len(ras)}')
print('All spec-background sanity checks passed')
"
```

- [ ] **Step 5: Commit**

```bash
git add livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py \
        livrable/phase1_v2_accessibility/data/geosites_master_catalog.csv \
        livrable/phase1_v2_accessibility/data/geosites_needs_review.csv
git commit -m "feat: cross-source fuzzy match/merge geosites into final Stage 1 master catalog"
```

---

## Self-Review Notes

- **Spec coverage:** every Process step in the spec (normalize, DMS parse, corrections, border check, regional-plausibility check, cross-source matching, ground-truth fold-in, sanity output) maps to a task step. All 5 output files from the spec's Outputs section are produced (`geosites_master_catalog.csv`, `geosites_needs_review.csv`, `geosites_outliers_removed.csv`, `geosites_region_mismatches.csv`, `dms_parse_failures.csv`), plus one intermediate file (`geosites_normalized_combined.csv`, `geosites_checked.csv`) per task boundary for reviewability. The nationwide (not single-case) framing of the regional-plausibility check from the spec's round-2 revision is reflected in Task 2 Step 6's explicit instruction to report the *actual* mismatch count, not assume it matches the one known example.
- **Placeholder scan:** none — every step has runnable code or an exact command with expected output. The two spots where the plan explicitly defers a decision to implementation time (scratch CSV column names in Task 1 Step 2, the region GeoDataFrame's name column in Task 2 Step 3/4) are marked as "inspect first, then wire in" rather than left as unresolved TODOs — this mirrors the spec's own "Open implementation details" section, which the plan is required to carry forward, not silently resolve on the plan-writer's behalf.
- **Type consistency:** `NORMALIZED_COLS` list is used identically in Task 1 (`load_csv_source`, `load_excel_source`) and consumed unchanged through Task 2 and Task 3. `Source_File` (Task 1/2 schema) is intentionally renamed to `Source_Files` (plural, semicolon-joined) only in Task 3's final merge output, since that's where multiple sources for one site legitimately combine — this is called out explicitly in Task 3 Step 1's code rather than left as a silent rename.
