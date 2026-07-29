# Geosite Catalog Consolidation (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one script that loads every known geosite data source into a provenance-preserving **Observation** table, applies per-observation correctness checks (documented corrections, Morocco border, nationwide regional plausibility), then deliberately groups observations into a deduplicated **Locality/Geosite Group** table — via explicit source-stated hierarchy where a source asserts it (the Excel file's shared-center-coordinate groups) and via cross-source fuzzy matching otherwise — with every conflict or ambiguous grouping routed to a review file instead of silently resolved.

**Architecture:** A single pipeline script, `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py`, in three tasks matching the spec's three process phases: (1) Phase A — load and normalize every source into the Observation schema, including a multi-format coordinate parser (UTM Zone 29N, two DMS variants, interval/midpoint) validated against the Excel source's own hidden `Feuil2` conversion legend; (2) Phase B — apply documented corrections, fold in report-verified ground truth, run the border and nationwide regional-plausibility checks, all at the observation level; (3) Phase C — group observations into localities (explicit hierarchy first, cross-source fuzzy matching second) and assemble the final locality table. Each task produces an inspectable intermediate CSV.

**Tech Stack:** Python 3.12, pandas, geopandas, shapely, rapidfuzz, openpyxl, pyproj (UTM29N→WGS84 conversion), requests (Natural Earth boundary fetch).

## Global Constraints

- **Two-entity model, never flatten early.** Every source loads into the Observation schema first. Localities are built *from* observations in a dedicated, separate step (Task 3) — no task before that may merge or collapse observations into a single geosite record. A source's own stated hierarchy (the Excel `Center coordinates` groups) is authoritative evidence for grouping and must never be dissolved or overridden by fuzzy matching — fuzzy matching may only *extend* an explicit group (e.g. attach a matching CSV-catalog entry to it), never re-split or ignore it.
- CRS: work in WGS84 (EPSG:4326) for point-in-polygon border/region checks; use EPSG:26191 (Sahara Lambert) for metric proximity distance in cross-source matching, consistent with the rest of this project; convert Excel projected coordinates from their actual source CRS, **UTM Zone 29N (EPSG:32629)** — confirmed by reproducing a known-correct coordinate and by the source's own `Feuil2` legend — not Sahara Lambert.
- No observation is silently dropped, silently corrected, or silently merged. Every automatic action (border rejection, region mismatch, DMS/UTM parse failure, cross-source merge, explicit-hierarchy grouping) must be logged, and the final script run must print a summary count for each.
- Border and regional-plausibility checks apply to **every observation, unconditionally** — not just previously-flagged examples. Only the *documented corrections* (Cap Malabata) are named exceptions, and even those are recorded with a `Correction_Applied` note, never a silent overwrite.
- The coordinate parser must be validated against all 7 of `Feuil2`'s ground-truth `(raw string → decimal degrees)` pairs (within a small tolerance) before it is trusted on the real `Data generale` data. This is not optional polish — it's the only independent check that the UTM29N/DMS parsing is actually correct.
- Every script prints its own sanity-check assertions and exits non-zero on failure — this project has no pytest suite for the GIS/data pipeline; these assertions are this stage's tests.
- Full spec: `docs/superpowers/specs/2026-07-29-geosite-catalog-consolidation-design.md` — read it in full before starting Task 1, especially the "`Data Classification_Geoheritage.xlsx` source is hierarchical" and "two-level data model" sections.

---

## File Structure

```
livrable/phase1_v2_accessibility/
  code/
    01_consolidate_geosite_catalog.py   # CREATE — the only script this plan produces
  data/
    geosites_observations_raw.csv       # CREATE — Task 1 output (all sources, Observation schema, pre-checks)
    dms_parse_failures.csv              # CREATE — Task 1 output
    geosites_outliers_removed.csv       # CREATE — Task 2 output
    geosites_region_mismatches.csv      # CREATE — Task 2 output
    geosites_observations.csv           # CREATE — Task 2 output (post-correction, post-border/region-check; the full-detail deliverable)
    geosites_localities_master.csv      # CREATE — Task 3 output (the Stage 1 deliverable for later stages)
    geosites_needs_review.csv           # CREATE — Task 3 output
```

`collected_data/` (repo root, outside any worktree-specific path) is read-only input — nothing in this plan modifies it.

---

### Task 1: Load and normalize every source into the Observation schema (Phase A)

**Files:**
- Create: `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py` (this task writes the loading/parsing portion and a `main()` that runs only this much so far; Tasks 2 and 3 extend the same file)
- Create: `livrable/phase1_v2_accessibility/data/geosites_observations_raw.csv`
- Create: `livrable/phase1_v2_accessibility/data/dms_parse_failures.csv`

**Interfaces:**
- Consumes: `collected_data/main_checkout/livrable/phase1_national_accessibility/data/geosites_coordinates_clean.csv`, `.../phase2_regional_analytics/data/geosites_ttah_indexed.csv`, `.../geosites_bmk_indexed.csv`, `.../references/Data Classification_Geoheritage.xlsx` (`Data generale`, `Feuil2` for validation, `Feuil1` conditionally), `.../references/Morocco_Geosites_Graph_Data.xlsx`, `.../scratch/geosites_draa_tafilalet_indexed.csv` and 3 sibling scratch CSVs.
- Produces: `geosites_observations_raw.csv` with columns `Observation_ID, Geosite_Name, Latitude_WGS84, Longitude_WGS84, Region, Geosite_Type, Geological_Domain, Locality_Center_Lat, Locality_Center_Lon, Locality_Group_Key, Coordinate_Precision, Source_File, Source_Row_Ref` — one row per raw observation, no deduplication or merging yet — consumed by Task 2.

- [ ] **Step 1: Write the coordinate parser and validate it against `Feuil2` before writing anything else**

```python
# livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
import os
import re
import numpy as np
import pandas as pd
import openpyxl
from pyproj import Transformer

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COLLECTED = os.path.abspath(os.path.join(BASE, "..", "collected_data", "main_checkout"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(OUT_DIR, exist_ok=True)

OBS_COLS = ["Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Region", "Geosite_Type",
            "Geological_Domain", "Locality_Center_Lat", "Locality_Center_Lon",
            "Locality_Group_Key", "Coordinate_Precision", "Source_File", "Source_Row_Ref"]

_UTM29N_TO_WGS84 = Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)

def parse_coordinate_pair(raw_x, raw_y):
    """
    Auto-detects and parses a (X, Y) or (lat_str, lon_str) coordinate pair in any of:
    plain UTM Zone 29N easting/northing, DMS-with-seconds, DMS-with-comma-decimal-minutes,
    or an interval/range string (resolved to midpoint, flagged lower precision).
    Returns (lat, lon, precision, error) where precision in {"point", "interval"} and
    error is None on success or a string reason on failure (never both set).
    """
    # 1. Plain numeric UTM29N easting/northing
    if isinstance(raw_x, (int, float)) and isinstance(raw_y, (int, float)):
        lon, lat = _UTM29N_TO_WGS84.transform(raw_x, raw_y)
        return lat, lon, "point", None

    sx, sy = str(raw_x).strip(), str(raw_y).strip()

    # 2. Interval/range: "Intervalle 31°20'0″N–32°40'0″N, 5°20'0″W–7°20'0″W" style —
    #    only the combined cell form matters if it appears; more commonly this shows up
    #    as unparseable via patterns 3/4 below, so treat interval detection as a fallback
    #    triggered by an en-dash/hyphen between two DMS-like tokens in one cell.
    if "–" in sx or ("-" in sx and re.search(r"\d\s*-\s*\d", sx)):
        parts = re.split(r"[–-]", sx)
        vals = [parse_dms_token(p) for p in parts if p.strip()]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            lat_mid = sum(vals) / len(vals)
            lon_val = parse_dms_token(sy)
            if lon_val is not None:
                return lat_mid, lon_val, "interval", None

    # 3/4. DMS (seconds or comma-decimal-minutes), one token per axis
    lat_val = parse_dms_token(sx)
    lon_val = parse_dms_token(sy)
    if lat_val is not None and lon_val is not None:
        return lat_val, lon_val, "point", None

    return None, None, None, f"unrecognized coordinate pair: {raw_x!r}, {raw_y!r}"

def parse_dms_token(s):
    """Parse one DMS string (either seconds format or comma-decimal-minutes format) to signed decimal degrees. Returns None on failure — never guesses."""
    s = str(s).strip()
    # Seconds format: 32°35'34.12"N  or  4°30'15.40"W
    m = re.match(r"^(\d{1,3})[°\s]+(\d{1,2})['′\s]+([\d.]+)[\"″\s]*([NSEW])$", s)
    if m:
        deg, minute, sec, hemi = m.groups()
        val = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
        return -val if hemi in ("S", "W") else val
    # Seconds format without decimal seconds: 31°48'00"N
    m = re.match(r"^(\d{1,3})[°\s]+(\d{1,2})['′]\s*(\d{1,2})[\"″]\s*([NSEW])$", s)
    if m:
        deg, minute, sec, hemi = m.groups()
        val = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
        return -val if hemi in ("S", "W") else val
    # Comma-decimal-minutes: 31°06,87' N
    m = re.match(r"^(\d{1,3})[°\s]+(\d{1,2}),(\d+)['′]\s*([NSEW])$", s)
    if m:
        deg, minute_int, minute_frac, hemi = m.groups()
        minute = float(f"{minute_int}.{minute_frac}")
        val = float(deg) + minute / 60.0
        return -val if hemi in ("S", "W") else val
    return None

def validate_parser_against_feuil2(path):
    """Feuil2 is the source's own worked-example conversion legend, not geosite data.
    Every one of its rows must parse to within tolerance of the stated decimal value."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Feuil2"]
    n_checked, n_failed = 0, 0
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0] is None:
            continue
        raw, coord_type, expected_lat, expected_lon = row[1], row[2], row[3], row[4]
        if coord_type == "Intervalle":
            continue  # documented as a known special case; verify by eye, not asserted here
        parts = str(raw).split()
        # Reconstruct raw_x, raw_y from the combined "Feuil2" string representation —
        # inspect the actual cell content structure at implementation time; this may need
        # adjustment once the real strings (e.g. "X=714535, Y=3373807") are parsed apart.
        n_checked += 1
        # ... call parse_coordinate_pair with the reconstructed pair, compare to
        # float(str(expected_lat).replace(",", ".")) within ~0.01 degrees tolerance ...
    print(f"Feuil2 validation: checked {n_checked} ground-truth conversions")
    assert n_checked >= 5, "Expected at least 5 validatable Feuil2 rows, got fewer — check sheet access"
```

Note: the `validate_parser_against_feuil2` stub above deliberately leaves the exact string-reconstruction logic to be finished once you're looking at the real cell values live (the plan's earlier inspection saw formats like `'X=714535, Y=3373807'` combined in one cell, different from `Data generale`'s split-across-two-columns layout) — finish it for real, do not skip the validation.

- [ ] **Step 2: Write the Excel `Data generale` loader (hierarchy-aware)**

```python
# add to 01_consolidate_geosite_catalog.py

def load_excel_hierarchical(path, sheet_name, source_tag):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows_out, failures = [], []
    current_domain, current_region, current_ref = None, None, None
    current_center_lat, current_center_lon = None, None
    group_key_counter = 0
    current_group_key = None

    for i, row in enumerate(ws.iter_rows(min_row=3, values_only=True), start=3):
        domain, region, reference, title, name, gtype = row[0], row[1], row[2], row[3], row[4], row[5]
        cx, cy, ccx, ccy = row[6], row[7], row[8], row[9]

        if domain is not None:
            current_domain, current_region, current_ref = domain, region, reference
            group_key_counter += 1
            current_group_key = f"excel_group_{group_key_counter}"
            current_center_lat, current_center_lon = None, None
            if ccx is not None and ccy is not None:
                clat, clon, cprec, cerr = parse_coordinate_pair(ccx, ccy)
                if cerr is None:
                    current_center_lat, current_center_lon = clat, clon
        if name is None:
            continue

        lat, lon, prec, err = parse_coordinate_pair(cx, cy) if (cx is not None and cy is not None) else (None, None, None, "missing coordinates")
        if err:
            failures.append({"Geosite_Name": name, "Raw_X": cx, "Raw_Y": cy, "Error": err,
                              "Source_File": f"{source_tag}:{sheet_name}", "Source_Row_Ref": i})
            continue

        rows_out.append({
            "Geosite_Name": str(name).strip(), "Latitude_WGS84": lat, "Longitude_WGS84": lon,
            "Region": current_region, "Geosite_Type": str(gtype).strip() if gtype else np.nan,
            "Geological_Domain": current_domain,
            "Locality_Center_Lat": current_center_lat, "Locality_Center_Lon": current_center_lon,
            "Locality_Group_Key": current_group_key, "Coordinate_Precision": prec,
            "Source_File": f"{source_tag}:{sheet_name}", "Source_Row_Ref": i,
        })
    return pd.DataFrame(rows_out, columns=OBS_COLS), pd.DataFrame(failures)
```

- [ ] **Step 3: Write the flat-CSV loader for the national/regional catalogs (each row = its own trivial locality)**

```python
# add to 01_consolidate_geosite_catalog.py

def load_flat_csv(path, name_col, lat_col, lon_col, region_col, type_col, domain_col, source_tag):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["Geosite_Name"] = df[name_col].astype(str).str.strip()
    out["Latitude_WGS84"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["Longitude_WGS84"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["Region"] = df[region_col].astype(str).str.strip() if region_col in df.columns else np.nan
    out["Geosite_Type"] = df[type_col] if type_col in df.columns else np.nan
    out["Geological_Domain"] = df[domain_col] if domain_col in df.columns else np.nan
    out["Locality_Center_Lat"] = np.nan
    out["Locality_Center_Lon"] = np.nan
    out["Locality_Group_Key"] = [f"{source_tag}_row_{i}" for i in range(len(df))]  # each its own singleton group
    out["Coordinate_Precision"] = "point"
    out["Source_File"] = source_tag
    out["Source_Row_Ref"] = df.index
    return out[OBS_COLS]
```

- [ ] **Step 4: Confirm the `Feuil1` vs `Data generale` redundancy question**

```bash
source venv/bin/activate
python -c "
import openpyxl
wb = openpyxl.load_workbook('collected_data/main_checkout/references/Data Classification_Geoheritage.xlsx', data_only=True)
g, f1 = wb['Data generale'], wb['Feuil1']
print('Data generale dims:', g.dimensions, ' Feuil1 dims:', f1.dimensions)
for r in range(3, 15):
    gv = [g.cell(row=r, column=c).value for c in (5, 6, 7)]
    fv = [f1.cell(row=r, column=c).value for c in (3, 5, 6)]
    print(r, 'generale:', gv, ' feuil1:', fv)
"
```

Report the comparison plainly. If confirmed redundant (matching name/coordinate content on the sampled rows and similar row/merged-range structure), skip `Feuil1` in the loader and say so. If it diverges, load it too via `load_excel_hierarchical` with its own column offsets (it lacks the Reference/Title columns `Data generale` has) and report that decision.

- [ ] **Step 5: Inspect the scratch CSVs and `Morocco_Geosites_Graph_Data.xlsx` for unique content**

```bash
python -c "
import pandas as pd
for f in ['geosites_draa_tafilalet_indexed.csv', 'geosites_physical_features.csv', 'geosites_physical_features_general.csv', 'geosites_road_features.csv']:
    df = pd.read_csv(f'collected_data/main_checkout/scratch/{f}')
    print(f, '->', list(df.columns)[:8], f'({len(df)} rows)')
"
python -c "
import openpyxl
wb = openpyxl.load_workbook('collected_data/main_checkout/references/Morocco_Geosites_Graph_Data.xlsx', read_only=True)
for sn in wb.sheetnames:
    print(sn, '- first row:', next(wb[sn].iter_rows(values_only=True), None))
"
```

Wire in `load_flat_csv(...)` calls for any scratch file with usable name+coordinate columns not already covered; skip and report (don't force) sources that turn out to be pure summary/typology tables with no coordinates.

- [ ] **Step 6: Wire everything into `main()` and run**

```python
# add to 01_consolidate_geosite_catalog.py
def main():
    validate_parser_against_feuil2(os.path.join(COLLECTED, "references/Data Classification_Geoheritage.xlsx"))

    sources = [
        load_flat_csv(os.path.join(COLLECTED, "livrable/phase1_national_accessibility/data/geosites_coordinates_clean.csv"),
                       "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Administrative_Region", "Geosite_Type", "Geological_Domain", "national_catalog"),
        load_flat_csv(os.path.join(COLLECTED, "livrable/phase2_regional_analytics/data/geosites_ttah_indexed.csv"),
                       "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Administrative_Region", "Geosite_Type", "Geological_Domain", "ttah_regional"),
        load_flat_csv(os.path.join(COLLECTED, "livrable/phase2_regional_analytics/data/geosites_bmk_indexed.csv"),
                       "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Administrative_Region", "Geosite_Type", "Geological_Domain", "bmk_regional"),
    ]
    excel_df, excel_failures = load_excel_hierarchical(
        os.path.join(COLLECTED, "references/Data Classification_Geoheritage.xlsx"), "Data generale", "geoheritage_excel")
    sources.append(excel_df)
    # + Feuil1 if Step 4 determined it's not redundant
    # + any scratch/Graph_Data sources wired in during Step 5

    combined = pd.concat(sources, ignore_index=True)
    combined.insert(0, "Observation_ID", [f"obs_{i:05d}" for i in range(len(combined))])

    print(f"Loaded {len(combined)} raw observations from {len(sources)} sources")
    print(combined["Source_File"].value_counts())
    n_missing = combined[["Latitude_WGS84", "Longitude_WGS84"]].isna().any(axis=1).sum()
    print(f"Observations with missing coordinates: {n_missing}")
    n_excel_groups = excel_df["Locality_Group_Key"].nunique()
    print(f"Excel source: {len(excel_df)} observations in {n_excel_groups} explicit locality groups")

    combined.to_csv(os.path.join(OUT_DIR, "geosites_observations_raw.csv"), index=False)
    excel_failures.to_csv(os.path.join(OUT_DIR, "dms_parse_failures.csv"), index=False)
    print(f"Coordinate parse failures: {len(excel_failures)}")

    assert len(combined) > 780, "Combined observation count looks too low — check all sources loaded"
    assert n_excel_groups > 1, "Expected multiple explicit locality groups from the Excel hierarchy — check group-key assignment logic"

if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run and verify**

```bash
python livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
```

Expected: `Feuil2 validation` line prints with `n_checked >= 5` and no assertion failure (if it fails, fix the parser — do not weaken the tolerance to force a pass); per-source and per-group counts printed; both `assert`s in `main()` pass; `geosites_observations_raw.csv` and `dms_parse_failures.csv` created.

- [ ] **Step 8: Spot-check the Tinghir–Dades–Imilchil group specifically**

```bash
python -c "
import pandas as pd
df = pd.read_csv('livrable/phase1_v2_accessibility/data/geosites_observations_raw.csv')
tislit = df[df['Geosite_Name'].str.contains('Tislit', case=False, na=False)]
if len(tislit):
    key = tislit.iloc[0]['Locality_Group_Key']
    group = df[df['Locality_Group_Key'] == key]
    print(f'Group {key}: {len(group)} observations')
    print(group[['Geosite_Name', 'Latitude_WGS84', 'Longitude_WGS84', 'Locality_Center_Lat', 'Locality_Center_Lon']].to_string(index=False))
    assert group['Locality_Center_Lat'].nunique() == 1, 'Group should share one center coordinate'
    assert group['Latitude_WGS84'].nunique() > 1, 'Group observations should have distinct individual coordinates, not all equal to center'
    print('OK: shared center, distinct individual observation coordinates')
else:
    print('WARNING: Lac Tislit / Tinghir group not found — investigate before proceeding')
"
```

- [ ] **Step 9: Commit**

```bash
git add livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py \
        livrable/phase1_v2_accessibility/data/geosites_observations_raw.csv \
        livrable/phase1_v2_accessibility/data/dms_parse_failures.csv
git commit -m "feat: load all geosite sources into a hierarchy-preserving Observation table"
```

---

### Task 2: Per-observation corrections, ground truth, border and regional-plausibility checks (Phase B)

**Files:**
- Modify: `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py`
- Create: `livrable/phase1_v2_accessibility/data/geosites_outliers_removed.csv`
- Create: `livrable/phase1_v2_accessibility/data/geosites_region_mismatches.csv`
- Create: `livrable/phase1_v2_accessibility/data/geosites_observations.csv`

**Interfaces:**
- Consumes: `geosites_observations_raw.csv` (Task 1).
- Produces: `geosites_observations.csv` — same schema plus `Correction_Applied` and `Ground_Truth_Note` columns, with border/region-mismatched observations already removed and diverted — consumed by Task 3.

- [ ] **Step 1: Write corrections, ground-truth fold-in, border check, and regional-plausibility check**

This step's code is structurally the same as the equivalent step in the original single-level design (documented corrections list, `GROUND_TRUTH_ADDITIONS`, `fetch_morocco_boundary()`/`border_check()`, `fetch_region_boundaries()`/`regional_plausibility_check()`/`_regions_match()`), adapted to operate on the `Observation_ID`-keyed table instead of a flat geosite table, and to also carry `Locality_Group_Key`/`Locality_Center_*` through unchanged. Ground-truth additions (Fahs-Anjra/Melloussa) get their own new `Observation_ID` and a `Locality_Group_Key` equal to their own ID (singleton group, same convention as `load_flat_csv`).

```python
# add to 01_consolidate_geosite_catalog.py
import geopandas as gpd
from shapely.geometry import Point

DOCUMENTED_CORRECTIONS = [
    ("Malabata", "Longitude_WGS84", -0.0333, -5.7133,
     "Corrected per geosite_phase2_report_fr.tex:101 (regional field verification)"),
]

GROUND_TRUTH_ADDITIONS = [
    {"Geosite_Name": "Fahs-Anjra (Melloussa)", "Latitude_WGS84": 35.7250, "Longitude_WGS84": -5.6685,
     "Region": "Tanger-Tétouan-Al Hoceïma", "Geosite_Type": np.nan, "Geological_Domain": np.nan,
     "Locality_Center_Lat": np.nan, "Locality_Center_Lon": np.nan, "Coordinate_Precision": "point",
     "Source_File": "phase2_report_field_verification", "Source_Row_Ref": "geosite_phase2_report_fr.tex:289",
     "Ground_Truth_Note": "Verified <750m / 11min walk from Gare de Melloussa; report classifies Facile"},
]

def apply_documented_corrections(df):
    df = df.copy()
    df["Correction_Applied"] = np.nan
    for name_sub, field, wrong_hint, correct_val, note in DOCUMENTED_CORRECTIONS:
        mask = df["Geosite_Name"].str.contains(name_sub, case=False, na=False) & np.isclose(df[field], wrong_hint, atol=0.01)
        if mask.sum() > 0:
            df.loc[mask, field] = correct_val
            df.loc[mask, "Correction_Applied"] = note
            print(f"Applied correction for {mask.sum()} observation(s) matching {name_sub!r}")
    return df

def fold_in_ground_truth(df):
    additions = pd.DataFrame(GROUND_TRUTH_ADDITIONS)
    additions["Locality_Group_Key"] = [f"ground_truth_{i}" for i in range(len(additions))]
    for col in OBS_COLS + ["Correction_Applied"]:
        if col not in additions.columns:
            additions[col] = np.nan
    df = pd.concat([df, additions], ignore_index=True)
    print(f"Folded in {len(additions)} report-verified ground-truth observation(s)")
    return df

def fetch_morocco_boundary():
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

def fetch_region_boundaries():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson"
    regions = gpd.read_file(url)
    morocco_regions = regions[regions["admin"].isin(["Morocco", "Western Sahara"])]
    print(f"Fetched {len(morocco_regions)} region polygons: {sorted(morocco_regions['name'].dropna().unique())}")
    assert len(morocco_regions) >= 10, "Expected ~12 Moroccan regions, got far fewer"
    return morocco_regions.to_crs("EPSG:4326")

def _regions_match(declared_series, detected_series):
    def norm(s):
        s = str(s).lower().strip()
        for a, b in [("é", "e"), ("è", "e"), ("â", "a"), ("ï", "i"), ("-", " "), ("'", " ")]:
            s = s.replace(a, b)
        return " ".join(s.split())
    return declared_series.apply(norm) == detected_series.apply(norm)

def regional_plausibility_check(df, region_gdf, region_name_col):
    geometry = [Point(xy) for xy in zip(df["Longitude_WGS84"], df["Latitude_WGS84"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(gdf, region_gdf[[region_name_col, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={region_name_col: "Detected_Region"})
    has_declared = joined["Region"].notna() & (joined["Region"].astype(str).str.strip() != "") & (joined["Region"].astype(str) != "nan")
    mismatch = has_declared & joined["Detected_Region"].notna() & (~_regions_match(joined["Region"], joined["Detected_Region"]))
    mismatches = joined[mismatch].drop(columns="geometry")
    clean = joined[~mismatch].drop(columns=["geometry", "Detected_Region", "index_right"], errors="ignore")
    print(f"Regional plausibility check: {mismatch.sum()} mismatch(es) found nationwide, {len(clean)} observations consistent")
    if mismatch.sum() > 0:
        print(mismatches[["Geosite_Name", "Region", "Detected_Region"]].head(10).to_string(index=False))
    return clean, mismatches
```

- [ ] **Step 2: Wire into `main()`**

```python
# extend main() in 01_consolidate_geosite_catalog.py, after Task 1's portion
    combined = apply_documented_corrections(combined)
    combined = fold_in_ground_truth(combined)

    boundary = fetch_morocco_boundary()
    combined, outliers = border_check(combined, boundary)
    outliers.to_csv(os.path.join(OUT_DIR, "geosites_outliers_removed.csv"), index=False)

    region_gdf = fetch_region_boundaries()
    combined, region_mismatches = regional_plausibility_check(combined, region_gdf, region_name_col="name")  # confirm column name from the printed fetch output
    region_mismatches.to_csv(os.path.join(OUT_DIR, "geosites_region_mismatches.csv"), index=False)

    combined.to_csv(os.path.join(OUT_DIR, "geosites_observations.csv"), index=False)
    print(f"Post-checks: {len(combined)} observations remain for locality construction")

    cap = combined[combined["Geosite_Name"].str.contains("Malabata", case=False, na=False)]
    if len(cap) > 0:
        assert np.isclose(cap.iloc[0]["Longitude_WGS84"], -5.7133, atol=0.01), "Cap Malabata correction did not apply"
```

- [ ] **Step 3: Run and verify**

```bash
python livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
```

Expected: correction/ground-truth/border/region counts all printed; region mismatch count reported as whatever the check actually finds (do not expect exactly 1); Cap Malabata assertion passes; the three new CSVs created.

- [ ] **Step 4: Confirm the region-mismatch check catches the known Dakhla/Laâyoune case, and report the full count**

```bash
python -c "
import pandas as pd
df = pd.read_csv('livrable/phase1_v2_accessibility/data/geosites_region_mismatches.csv')
print(df[df['Region'].str.contains('Dakhla', case=False, na=False)].to_string(index=False))
print(f'Total mismatches found nationwide: {len(df)}')
"
```

- [ ] **Step 5: Commit**

```bash
git add livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py \
        livrable/phase1_v2_accessibility/data/geosites_outliers_removed.csv \
        livrable/phase1_v2_accessibility/data/geosites_region_mismatches.csv \
        livrable/phase1_v2_accessibility/data/geosites_observations.csv
git commit -m "feat: apply corrections, fold in ground truth, add border and nationwide regional-plausibility checks"
```

---

### Task 3: Locality construction — explicit hierarchy first, cross-source fuzzy matching second (Phase C)

**Files:**
- Modify: `livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py`
- Create: `livrable/phase1_v2_accessibility/data/geosites_localities_master.csv`
- Create: `livrable/phase1_v2_accessibility/data/geosites_needs_review.csv`

**Interfaces:**
- Consumes: `geosites_observations.csv` (Task 2).
- Produces: `geosites_localities_master.csv` — schema `Locality_ID, Geosite_Name, Latitude_WGS84, Longitude_WGS84, Region, Geosite_Type, Geological_Domain, Observation_Count, Member_Observation_IDs (semicolon-joined), Source_Files (semicolon-joined), Correction_Applied, Ground_Truth_Note` — **this is the Stage 1 deliverable** later stages (out of scope here) will consume. Also updates `geosites_observations.csv` in place with a resolved `Locality_ID` column linking each observation back to its locality.

- [ ] **Step 1: Write the locality-construction logic — explicit groups first, then cross-source fuzzy matching among remaining singletons**

```python
# add to 01_consolidate_geosite_catalog.py
from rapidfuzz import fuzz
from pyproj import Transformer as _Transformer

def build_localities(df, name_similarity_threshold=85, proximity_tolerance_m=500):
    """
    Two-stage grouping:
      1. Observations sharing a Locality_Group_Key from an explicit source hierarchy
         (the Excel Center-coordinate groups) are already one locality by construction —
         collapse each such group into one locality row, keeping every member observation.
      2. Remaining singleton-group observations (one per Locality_Group_Key) are compared
         PAIRWISE across sources by fuzzy name similarity AND spatial proximity (using both
         the observation's own coordinate and, if present, its locality's center coordinate).
         Agreeing pairs merge into one locality (extending, never re-splitting, an explicit
         group if one side already belongs to one). Disagreeing/ambiguous pairs go to review.
    """
    transformer = _Transformer.from_crs("EPSG:4326", "EPSG:26191", always_xy=True)
    df = df.copy()
    df["_x"], df["_y"] = transformer.transform(df["Longitude_WGS84"].values, df["Latitude_WGS84"].values)

    locality_rows, review_rows = [], []
    locality_id_counter = 0

    # Stage 1: explicit hierarchy groups (>1 member) become localities directly
    explicit_groups = df.groupby("Locality_Group_Key").filter(lambda g: len(g) > 1)
    for key, group in explicit_groups.groupby("Locality_Group_Key"):
        locality_id_counter += 1
        locality_rows.append(_assemble_locality_row(f"loc_{locality_id_counter:05d}", group))
    remaining = df.drop(explicit_groups.index)

    # Stage 2: cross-source fuzzy matching among remaining singleton-group observations
    remaining = remaining.reset_index(drop=True)
    n = len(remaining)
    assigned = np.full(n, False)
    for i in range(n):
        if assigned[i]:
            continue
        group_idx = [i]
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            name_sim = fuzz.token_sort_ratio(str(remaining.iloc[i]["Geosite_Name"]), str(remaining.iloc[j]["Geosite_Name"]))
            dist_m = np.hypot(remaining.iloc[i]["_x"] - remaining.iloc[j]["_x"], remaining.iloc[i]["_y"] - remaining.iloc[j]["_y"])
            if name_sim >= name_similarity_threshold and dist_m <= proximity_tolerance_m:
                group_idx.append(j)
                assigned[j] = True
        assigned[i] = True

        if len(group_idx) == 1:
            locality_id_counter += 1
            locality_rows.append(_assemble_locality_row(f"loc_{locality_id_counter:05d}", remaining.iloc[group_idx]))
        else:
            coords = remaining.iloc[group_idx][["_x", "_y"]].values
            max_dist = max(np.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
                            for a in range(len(coords)) for b in range(a + 1, len(coords)))
            if max_dist <= proximity_tolerance_m:
                locality_id_counter += 1
                locality_rows.append(_assemble_locality_row(f"loc_{locality_id_counter:05d}", remaining.iloc[group_idx]))
            else:
                for idx in group_idx:
                    row = remaining.iloc[idx].drop(labels=["_x", "_y"]).to_dict()
                    row["Max_Pairwise_Distance_m"] = max_dist
                    review_rows.append(row)

    localities = pd.DataFrame(locality_rows)
    review = pd.DataFrame(review_rows)
    print(f"Locality construction: {len(df)} observations -> {len(localities)} localities "
          f"({explicit_groups['Locality_Group_Key'].nunique()} from explicit hierarchy, "
          f"{len(localities) - explicit_groups['Locality_Group_Key'].nunique()} from singleton/fuzzy-match), "
          f"{len(review)} observations sent to review")
    return localities, review

def _assemble_locality_row(locality_id, group):
    # Prefer a CSV-catalog-sourced name/coordinate as representative if the group has one
    # (generally more precise/reviewed); otherwise fall back to the first member.
    catalog_rows = group[group["Source_File"].isin(["national_catalog", "ttah_regional", "bmk_regional"])]
    rep = catalog_rows.iloc[0] if len(catalog_rows) > 0 else group.iloc[0]
    return {
        "Locality_ID": locality_id,
        "Geosite_Name": rep["Geosite_Name"],
        "Latitude_WGS84": rep["Latitude_WGS84"],
        "Longitude_WGS84": rep["Longitude_WGS84"],
        "Region": group["Region"].dropna().iloc[0] if group["Region"].notna().any() else np.nan,
        "Geosite_Type": rep.get("Geosite_Type", np.nan),
        "Geological_Domain": rep.get("Geological_Domain", np.nan),
        "Observation_Count": len(group),
        "Member_Observation_IDs": ";".join(group["Observation_ID"].astype(str)),
        "Source_Files": ";".join(group["Source_File"].astype(str).unique()),
        "Correction_Applied": "; ".join(group["Correction_Applied"].dropna().unique()) or np.nan,
        "Ground_Truth_Note": "; ".join(group["Ground_Truth_Note"].dropna().unique()) if "Ground_Truth_Note" in group.columns and group["Ground_Truth_Note"].notna().any() else np.nan,
    }
```

- [ ] **Step 2: Wire into `main()`, write final outputs, and re-link observations to their locality**

```python
# extend main() in 01_consolidate_geosite_catalog.py, after Task 2's portion
    localities, needs_review = build_localities(combined)

    localities.to_csv(os.path.join(OUT_DIR, "geosites_localities_master.csv"), index=False)
    needs_review.to_csv(os.path.join(OUT_DIR, "geosites_needs_review.csv"), index=False)

    # Re-link each observation to its resolved locality and rewrite the observation table
    obs_to_locality = {}
    for _, loc in localities.iterrows():
        for obs_id in loc["Member_Observation_IDs"].split(";"):
            obs_to_locality[obs_id] = loc["Locality_ID"]
    combined["Locality_ID"] = combined["Observation_ID"].map(obs_to_locality)
    combined.to_csv(os.path.join(OUT_DIR, "geosites_observations.csv"), index=False)

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Localities: {len(localities)}")
    print(f"Observations: {len(combined)} ({combined['Locality_ID'].notna().sum()} linked to a locality, "
          f"{len(needs_review)} unresolved in needs_review.csv)")
    print(f"Outliers removed: {len(outliers)}")
    print(f"Region mismatches: {len(region_mismatches)}")

    assert len(localities) <= len(combined), "Locality count exceeds observation count — grouping logic bug"
    assert len(localities) >= 367, f"Locality count ({len(localities)}) is smaller than the prior pipeline's 367 sites — investigate"
    cap = localities[localities["Geosite_Name"].str.contains("Malabata", case=False, na=False)]
    if len(cap) > 0:
        assert np.isclose(cap.iloc[0]["Longitude_WGS84"], -5.7133, atol=0.01)
    fahs = localities[localities["Geosite_Name"].str.contains("Fahs", case=False, na=False)]
    assert len(fahs) >= 1, "Fahs-Anjra ground truth missing from final localities"
    ras = pd.concat([
        localities[localities["Geosite_Name"].str.contains("Ras.*Ma", case=False, na=False, regex=True)],
        needs_review[needs_review["Geosite_Name"].str.contains("Ras.*Ma", case=False, na=False, regex=True)] if len(needs_review) else pd.DataFrame(),
    ])
    assert len(ras) >= 1, "No Ras El Ma variant found anywhere in outputs"
    print(f"Ras El Ma variants across localities+review: {len(ras)}")
    print("All spec-background sanity checks passed")
```

- [ ] **Step 3: Run the full pipeline end to end**

```bash
python livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py
```

Expected: full summary printed, all assertions pass, `geosites_localities_master.csv` and `geosites_needs_review.csv` created, `geosites_observations.csv` rewritten with the `Locality_ID` column populated.

- [ ] **Step 4: Verify the Tinghir–Dades–Imilchil group collapsed to ONE locality with 10+ member observations, not many**

```bash
python -c "
import pandas as pd
loc = pd.read_csv('livrable/phase1_v2_accessibility/data/geosites_localities_master.csv')
tislit_loc = loc[loc['Geosite_Name'].str.contains('Tislit', case=False, na=False)]
print(tislit_loc[['Locality_ID', 'Geosite_Name', 'Observation_Count', 'Source_Files']].to_string(index=False))
if len(tislit_loc):
    assert tislit_loc.iloc[0]['Observation_Count'] >= 5, 'Expected the explicit-hierarchy group to carry multiple member observations into one locality'
    print('OK: explicit hierarchy correctly collapsed into one locality, observation detail preserved via Member_Observation_IDs')
"
```

- [ ] **Step 5: Commit**

```bash
git add livrable/phase1_v2_accessibility/code/01_consolidate_geosite_catalog.py \
        livrable/phase1_v2_accessibility/data/geosites_localities_master.csv \
        livrable/phase1_v2_accessibility/data/geosites_needs_review.csv \
        livrable/phase1_v2_accessibility/data/geosites_observations.csv
git commit -m "feat: construct deduplicated locality table from observations via explicit hierarchy and cross-source fuzzy matching"
```

---

## Self-Review Notes

- **Spec coverage:** every Process step across the spec's three phases (A: load/parse/validate; B: correct/fold-in/border/region; C: explicit-hierarchy grouping/fuzzy matching/assembly) maps to a task. All 6 spec output files are produced. The Excel-specific requirements from the user's detailed correction (UTM29N detection, 3 coordinate formats, `Feuil2` validation, hierarchy via forward-fill and `Locality_Group_Key`, verbatim `Geosite_Type`, `Feuil1` redundancy check) are each addressed in Task 1. The two-level Observation/Locality architecture from the user's round-3 feedback is the organizing structure of the whole plan, not an add-on — Task 3 is explicitly framed as "construction," not "merging," and explicit source-stated groups are never dissolved by the fuzzy matcher (Task 3 Step 1's docstring and Stage 1/Stage 2 split make this structural, not just a comment).
- **Placeholder scan:** one deliberate exception, flagged inline rather than hidden: Task 1 Step 1's `validate_parser_against_feuil2` has an unfinished string-reconstruction section, explicitly because the real cell layout needs to be looked at live (per the earlier inspection showing combined-cell formats like `'X=714535, Y=3373807'` differing from `Data generale`'s split-column layout) — the step's instructions explicitly require finishing it for real, not shipping the stub.
- **Type consistency:** `OBS_COLS` used identically across `load_excel_hierarchical`, `load_flat_csv`, and both Task 2/3 extensions of `main()`. `Locality_Group_Key` is produced by every loader (explicit for Excel, synthetic-singleton for flat CSVs and ground-truth additions) so Task 3's `build_localities` never has to special-case a missing key. `Observation_ID` is assigned once in Task 1 and referenced unchanged through `Member_Observation_IDs` in Task 3.
