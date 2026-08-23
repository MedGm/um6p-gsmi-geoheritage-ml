"""
Ingests the expanded Aug-9-2026 geoheritage literature source (references/new-db/) using
the same Observation/Locality pipeline already built in _catalog_helpers.py (originally code/01_consolidate_geosite_catalog.py, renamed
during the 2026-08-23 repo cleanup since it's imported as a library, not run directly)
-- reusing its coordinate parser, UTM zone handling (29N default, 28N Dakhla override,
Merchich/Nord Maroc Lambert grid for the Chefchaouen park group), ground-truth
corrections, border check, regional-plausibility check, and locality construction
verbatim (imported, not reimplemented).

Why a new script instead of editing _catalog_helpers.py directly: it
points at collected_data/main_checkout/references/Data Classification_Geoheritage.xlsx
(dated 2026-07-29, .xlsx, read via openpyxl) -- confirmed to have only 766 "Data
generale" observation rows. The user's Aug-9 delivery (references/new-db/, exported as
CSV, not .xlsx) is the SAME underlying document expanded to 1465 rows (same lineage:
identical first 8 site names, identical known-correct UTM anchor point) -- this is the
"new database with more studied geosites" the supervisors described. Since the input
changed shape (CSV, not .xlsx; different row alignment) and 01 was never run (no
outputs exist, file is untracked in git), this script adapts the *input loading* layer
only and imports every parsing/checking/matching function from 01 unchanged.

Two things intentionally NOT carried over:
- "Nouvelles données" (the 3rd new-db CSV) is NOT ingested as a separate source: matched
  against Data generale first (scratch/newdb_matching.py, same session) and found to be
  a 99%+ duplicate (651/889 exact name+coord matches, 232 more name-only) -- ingesting
  it separately would double-count nearly every observation. Documented, not silently
  dropped.
- No Feuil1-equivalent CSV was provided in this delivery (only Data generale, Nouvelles
  données, and the coordinate-conversion legend). 01's Feuil1 fallback fills in a
  handful of groups' *Center coordinates* when Data generale's own is a placeholder --
  a minor enhancement, not used by individual-observation coordinates or by locality
  clustering (which uses each observation's own coordinate, never the group center,
  per 01's 2026-07-30 finding). Its absence here is low-impact and explicitly noted in
  the run summary below, not silently ignored.
"""
import importlib.util
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
NEWDB = os.path.join(BASE, "references", "databases", "new-db")
OUT_DIR = os.path.join(BASE, "data", "newdb_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# --- import _catalog_helpers.py's proven functions (kept as a library module, not run directly) ---
spec = importlib.util.spec_from_file_location("consolidate01", os.path.join(HERE, "_catalog_helpers.py"))
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)

GEN_CSV = os.path.join(NEWDB, "Data Classification_09-08-2026(Data generale).csv")
CONV_CSV = os.path.join(NEWDB, "Data Classification_09-08-2026(Convertion des coordonées).csv")


# ---------------------------------------------------------------------------
# Adapter 1: read Data generale's CSV export as a raw (un-forward-filled) row
# iterable in the exact 10-column order load_hierarchical_rows expects
# (domain, region, reference, title, name, gtype, cx, cy, ccx, ccy) -- mirrors
# openpyxl's ws.iter_rows(min_row=3, values_only=True) shape exactly, so the
# group-boundary logic ("new group starts when domain is non-null") behaves
# identically to the xlsx path.
# ---------------------------------------------------------------------------
def load_general_csv_rows(path):
    df = pd.read_csv(path, encoding="utf-8-sig", header=[0, 1])
    df.columns = ["_".join([str(a) for a in col if "Unnamed" not in str(a)]) for col in df.columns]
    df = df.dropna(axis=1, how="all")
    # disambiguate the two 'Y_Northing' columns (Coordinates block vs Center coordinates block)
    cols = list(df.columns)
    y_idx = [i for i, c in enumerate(cols) if c == "Y_Northing"]
    cols[y_idx[0]] = "Coord_Y_Northing"
    cols[y_idx[1]] = "Center_Y_Northing"
    df.columns = cols
    df = df.rename(columns={"Coordinates_X_Easting": "Coord_X_Easting"})

    def clean(v):
        if pd.isna(v):
            return None
        return v

    def fix_spurious_letter_utm(cx, cy):
        """The CSV export adds a spurious trailing N/W letter to some plain-numeric UTM
        cells (e.g. '680098N'/'3380902W' for a real UTM pair) that the xlsx source stored
        as pure numbers -- this collides with parse_coordinate_pair's Merchich/Lambert-grid
        regex (also digits+N/digits+W, but much smaller magnitude: Morocco's Merchich
        northing is ~200k-600k, vs a real UTM northing of ~2.0M-4.2M). Distinguish by the
        Northing value's magnitude and strip the letter + return numerics for the real-UTM
        case, restoring the xlsx path's original numeric-cell (branch-1) behavior. Leaves
        genuine small-magnitude Merchich-shaped pairs (and everything else) untouched."""
        if not (isinstance(cx, str) and isinstance(cy, str)):
            return cx, cy
        mx = re.match(r'^(\d{5,7})N$', cx)
        my = re.match(r'^(\d{5,7})W$', cy)
        if mx and my and float(my.group(1)) >= 1_000_000:
            return float(mx.group(1)), float(my.group(1))
        return cx, cy

    rows = []
    for i, row in df.iterrows():
        cx, cy = fix_spurious_letter_utm(clean(row["Coord_X_Easting"]), clean(row["Coord_Y_Northing"]))
        ccx, ccy = fix_spurious_letter_utm(clean(row["Center coordinates_X_Easting"]), clean(row["Center_Y_Northing"]))
        rows.append((
            i + 3,  # row index, matching the xlsx convention (data starts at row 3)
            (
                clean(row["Geological domain"]), clean(row["Administrative region"]),
                clean(row["Reference"]), clean(row["Title"]), clean(row["Evaluated geosites"]),
                clean(row["Geosite's type"]), cx, cy, ccx, ccy,
            ),
        ))
    return rows


# ---------------------------------------------------------------------------
# Adapter 2: the "Convertion des coordonées" CSV's real legend columns are
# ID / Coordonnées originales / Type / Latitude / Longitude / Remarques
# (columns 8-13 of the raw export -- columns 0-7 are a stray misaligned copy
# of Data generale-style columns, confirmed unreliable this session: only 26
# of ~1130 rows have any ID at all). Extract just the legend block, matching
# openpyxl row[0..5] shape (id, raw, type, expected_lat, expected_lon, remarks).
# ---------------------------------------------------------------------------
def load_legend_csv_rows(path):
    df = pd.read_csv(path, encoding="utf-8-sig", header=[0, 1])
    df.columns = ["_".join([str(a) for a in col if "Unnamed" not in str(a)]) for col in df.columns]
    df = df.dropna(axis=1, how="all")
    legend = df[df["ID"].notna()].copy()
    rows = []
    for _, row in legend.iterrows():
        rows.append((
            row["ID"], row["Coordonnées originales"], row["Type"],
            row["Coordonées centrique_Latitude (° déc.)"], row["Longitude (° déc.)"], row["Remarques"],
        ))
    return rows


# ---------------------------------------------------------------------------
# Row-source versions of 01's openpyxl-bound loader/validator functions.
# Logic copied verbatim from 01 (same conditions, same UTM zone lookup, same
# Merchich plausibility check) -- only the row source changed from
# `ws.iter_rows(...)` to a plain Python list already built above.
# ---------------------------------------------------------------------------
def validate_legend_against_source_rows(legend_rows):
    """Same intent as 01's validate_parser_against_feuil2, adapted to the CSV legend's
    26 rows (vs. the xlsx's original 7-26 depending on vintage). Same conclusion
    expected: DMS-family rows should parse tightly; UTM rows are known-imprecise in
    this legend (see 01's docstring) and are only aggregate-checked, not per-row."""
    n_checked, n_failed = 0, 0
    utm_rows = []
    for id_, raw, coord_type, exp_lat, exp_lon, remarks in legend_rows:
        try:
            exp_lat_f = c01._to_float_expected(exp_lat)
            exp_lon_f = c01._to_float_expected(str(exp_lon).replace("?", "-"))
        except (TypeError, ValueError):
            continue
        if coord_type == "UTM":
            x, y = c01.extract_utm_xy_from_string(str(raw))
            if x is None:
                continue
            utm_rows.append((x, y, exp_lat_f, exp_lon_f))
            continue
        if coord_type != "DMS":
            continue
        tokens = re.findall(r"[\d][\d°'’′‘\"″ʺ”“,.\s]*[NSEW]", str(raw))
        lat_tok = next((t for t in tokens if t.strip()[-1] in "NS"), None)
        lon_tok = next((t for t in tokens if t.strip()[-1] in "EW"), None)
        if lat_tok is None or lon_tok is None:
            continue
        lat, lon, prec, err = c01.parse_coordinate_pair(lat_tok.strip(), lon_tok.strip())
        n_checked += 1
        if err is not None or lat is None or abs(lat - exp_lat_f) > 0.5 or abs(lon - exp_lon_f) > 0.5:
            n_failed += 1
            print(f"  LEGEND WARNING id={id_} raw={raw!r} expected=({exp_lat_f},{exp_lon_f}) got=({lat},{lon}) err={err}", flush=True)
    print(f"Legend spot-check (DMS-family, loose 0.5deg tolerance since this legend's own values "
          f"are hand-estimates beyond the first few rows): checked {n_checked}, {n_failed} outside tolerance", flush=True)
    if len(utm_rows) >= 3:
        zone_errors = {}
        for zone in (28, 29, 30):
            total = sum(abs(c01.parse_coordinate_pair(x, y, utm_zone=zone)[0] - elat) +
                        abs(c01.parse_coordinate_pair(x, y, utm_zone=zone)[1] - elon)
                        for x, y, elat, elon in utm_rows)
            zone_errors[zone] = total
        print(f"Legend UTM aggregate error by zone: {zone_errors}", flush=True)


_MERCHICH_ROW_RE_X = re.compile(r'^(\d{5,7})N$')
_MERCHICH_ROW_RE_Y = re.compile(r'^(\d{5,7})W$')


def load_hierarchical_rows(row_source, source_tag, feuil2_overrides=None):
    """Same grouping/parsing logic as 01's load_excel_hierarchical, driven by a plain
    (row_idx, row_tuple) list instead of an openpyxl worksheet. feuil1_centers is
    always {} here (no Feuil1-equivalent in this delivery, see module docstring) --
    every observation whose raw coordinate matches the Merchich/Nord Maroc bare-digit
    shape is tagged Is_Merchich_Format so post_hoc_merchich_check() (below) can apply
    the group's own computed median as a fallback plausibility center, since the
    per-group declared center (01's normal safeguard for this checksum-free format) is
    unavailable without Feuil1."""
    feuil2_overrides = feuil2_overrides or []
    rows_out, failures = [], []
    current_domain, current_region, current_ref = None, None, None
    current_center_lat, current_center_lon, current_center_source = None, None, None
    group_key_counter = 0
    current_group_key = None

    for i, row in row_source:
        domain, region, reference, title, name, gtype = row[0], row[1], row[2], row[3], row[4], row[5]
        cx, cy, ccx, ccy = row[6], row[7], row[8], row[9]

        if domain is not None:
            current_domain, current_region, current_ref = domain, region, reference
            group_key_counter += 1
            current_group_key = f"{source_tag}_group_{group_key_counter}"
            current_center_lat, current_center_lon, current_center_source = None, None, None
            zone = c01._group_utm_zone(reference)

            if ccx is not None and ccy is not None and not c01._is_placeholder_coord(ccx) and not c01._is_placeholder_coord(ccy):
                clat, clon, cprec, cerr = c01.parse_coordinate_pair(ccx, ccy, utm_zone=zone)
                if cerr is None:
                    current_center_lat, current_center_lon = clat, clon
                    current_center_source = "Data_generale"
                    olat, olon, oraw = c01._apply_feuil2_override(clat, clon, feuil2_overrides)
                    if oraw is not None:
                        current_center_lat, current_center_lon = olat, olon
                        current_center_source = "legend_lookup"
        if name is None:
            continue

        lat, lon, prec, err = c01.parse_coordinate_pair(
            cx, cy, locality_center=(current_center_lat, current_center_lon)
        ) if cx is not None else (None, None, None, "missing coordinates")
        if err:
            failures.append({"Geosite_Name": name, "Raw_X": cx, "Raw_Y": cy, "Error": err,
                              "Source_File": source_tag, "Source_Row_Ref": i})
            continue

        is_merchich = bool(isinstance(cx, str) and isinstance(cy, str) and
                           _MERCHICH_ROW_RE_X.match(cx) and _MERCHICH_ROW_RE_Y.match(cy))
        rows_out.append({
            "Geosite_Name": str(name).strip(), "Latitude_WGS84": lat, "Longitude_WGS84": lon,
            "Region": current_region, "Geosite_Type": str(gtype).strip() if gtype else np.nan,
            "Geological_Domain": current_domain,
            "Locality_Center_Lat": current_center_lat, "Locality_Center_Lon": current_center_lon,
            "Locality_Group_Key": current_group_key, "Coordinate_Precision": prec,
            "Source_File": source_tag, "Source_Row_Ref": i,
            "Center_Coord_Source": current_center_source,
            "Is_Merchich_Format": is_merchich,
        })
    return pd.DataFrame(rows_out, columns=c01.OBS_COLS_EXCEL + ["Is_Merchich_Format"]), pd.DataFrame(failures)


def post_hoc_merchich_check(df, max_km_from_group_median=50.0):
    """Fallback safeguard for the checksum-free Merchich/Nord Maroc format when no
    per-group declared center is available (this delivery has no Feuil1 CSV, so 01's
    normal per-row center-distance check silently no-ops for these groups -- confirmed
    2026-08-09: the 34-member Talassemtane/Chefchaouen park group has zero declared
    centers, and 3 of its members -- 'Beni Derkoul Radiolarites', 'Rueda Nummilitic
    succession', 'Jbel Lakraa Panorama', all independently confirmed corrupted/wrong-
    region by user review or already named in 01's own code comments -- slipped through
    as real-looking but ~400km-off coordinates). For each locality group with >=5
    members, compute the group's own median lat/lon from ALL its successfully-parsed
    members (a robust reference regardless of format), then flag any Is_Merchich_Format
    member farther than max_km_from_group_median from it -- same 50km threshold 01
    already established as correctly separating genuine within-group spread from
    corrupted-digit outliers."""
    if "Is_Merchich_Format" not in df.columns or not df["Is_Merchich_Format"].any():
        return df, pd.DataFrame(columns=df.columns)
    flagged_idx = []
    for key, group in df.groupby("Locality_Group_Key"):
        coord_ok = group["Latitude_WGS84"].notna() & group["Longitude_WGS84"].notna()
        if coord_ok.sum() < 5:
            continue
        med_lat, med_lon = group.loc[coord_ok, "Latitude_WGS84"].median(), group.loc[coord_ok, "Longitude_WGS84"].median()
        merchich_members = group[group["Is_Merchich_Format"] & coord_ok]
        for idx, row in merchich_members.iterrows():
            dist = c01._haversine_km(row["Latitude_WGS84"], row["Longitude_WGS84"], med_lat, med_lon)
            if dist > max_km_from_group_median:
                flagged_idx.append(idx)
                print(f"  Post-hoc Merchich check: '{row['Geosite_Name']}' is {dist:.0f}km from its "
                      f"group's own median ({med_lat:.4f},{med_lon:.4f}) -- exceeds {max_km_from_group_median:.0f}km, "
                      f"flagged as likely corrupted/truncated input digits", flush=True)
    flagged = df.loc[flagged_idx].copy()
    kept = df.drop(index=flagged_idx)
    print(f"Post-hoc Merchich check: {len(flagged)} observation(s) flagged and removed "
          f"(no declared group center was available to catch these at parse time)", flush=True)
    return kept, flagged


def build_legend_overrides_from_rows(legend_rows):
    """Same intent as 01's build_feuil2_center_overrides, from the CSV legend rows."""
    overrides = []
    for id_, raw, coord_type, exp_lat, exp_lon, remarks in legend_rows:
        if coord_type not in ("DMS", "Intervalle"):
            continue
        try:
            exp_lat_f = c01._to_float_expected(exp_lat)
            exp_lon_f = c01._to_float_expected(str(exp_lon).replace("?", "-"))
        except (TypeError, ValueError):
            continue
        if coord_type == "Intervalle":
            lat, lon, prec, err = c01.parse_combined_interval_with_hemisphere(str(raw))
        else:
            tokens = re.findall(r"[\d][\d°'’′‘\"″ʺ”“,.\s]*[NSEW]", str(raw))
            lat_tok = next((t for t in tokens if t.strip()[-1] in "NS"), None)
            lon_tok = next((t for t in tokens if t.strip()[-1] in "EW"), None)
            if lat_tok is None or lon_tok is None:
                continue
            lat, lon, prec, err = c01.parse_coordinate_pair(lat_tok.strip(), lon_tok.strip())
        if err is not None or lat is None:
            continue
        overrides.append((lat, lon, exp_lat_f, exp_lon_f, raw))
    return overrides


def main():
    print("=== Step 0: independent UTM anchor validation (same known-correct point as 01) ===", flush=True)
    c01.validate_utm_against_known_anchor()

    print("\n=== Step 1: load legend + validate ===", flush=True)
    legend_rows = load_legend_csv_rows(CONV_CSV)
    print(f"Legend rows (real, ID-tagged): {len(legend_rows)}", flush=True)
    validate_legend_against_source_rows(legend_rows)
    legend_overrides = build_legend_overrides_from_rows(legend_rows)
    print(f"Legend DMS/Interval lookup overrides available: {len(legend_overrides)}", flush=True)

    print("\n=== Step 2: load Data generale (expanded, 2026-08-09) ===", flush=True)
    general_rows = load_general_csv_rows(GEN_CSV)
    excel_df, excel_failures = load_hierarchical_rows(general_rows, "geoheritage_expanded_20260809", feuil2_overrides=legend_overrides)
    n_legend_used = (excel_df["Center_Coord_Source"] == "legend_lookup").sum()
    print(f"Parsed {len(excel_df)} observations ({excel_df['Latitude_WGS84'].notna().sum()} with coordinates), "
          f"{len(excel_failures)} coordinate parse failures, {n_legend_used} used the legend override", flush=True)
    print(f"NOTE: no Feuil1-equivalent CSV in this delivery -- Center coordinates fallback for "
          f"placeholder ('….') groups is unavailable this run (see module docstring); low impact, "
          f"locality clustering uses each observation's own coordinate, never the group center.", flush=True)

    n_groups = excel_df["Locality_Group_Key"].nunique()
    print(f"Explicit locality groups (publication-level, NOT physical-identity -- per 01's finding): {n_groups}", flush=True)

    combined = excel_df.copy()
    combined.insert(0, "Observation_ID", [f"obs2_{i:05d}" for i in range(len(combined))])
    combined.to_csv(os.path.join(OUT_DIR, "geosites_observations_raw.csv"), index=False)
    excel_failures.to_csv(os.path.join(OUT_DIR, "dms_parse_failures.csv"), index=False)

    assert len(combined) > 1000, "Expected >1000 observations from the expanded source — check loader"
    assert n_groups > 50, "Expected many explicit locality groups — check group-key assignment"

    print("\n=== Step 3: corrections, ground truth, border + regional checks ===", flush=True)
    combined = c01.apply_documented_corrections(combined)
    combined = c01.apply_ground_truth_notes(combined)
    # fold_in_ground_truth() is 01-catalog-specific (Fahs-Anjra etc, already exist in the
    # production catalog) -- not re-added here, this source is additive material only.

    boundary = c01.fetch_morocco_boundary()
    combined, outliers = c01.border_check(combined, boundary)
    outliers.to_csv(os.path.join(OUT_DIR, "geosites_outliers_removed.csv"), index=False)

    # Runs before the region check, not after: these are corrupted-coordinate rows
    # (wrong lat/lon), not genuine region-mislabeling -- if run after, they'd get
    # miscategorized into geosites_region_mismatches.csv instead of being cleanly
    # caught as the coordinate-corruption they actually are.
    combined, merchich_flagged = post_hoc_merchich_check(combined)
    merchich_flagged.to_csv(os.path.join(OUT_DIR, "geosites_merchich_posthoc_outliers.csv"), index=False)

    region_gdf = c01.fetch_region_boundaries()
    combined, region_mismatches = c01.regional_plausibility_check(combined, region_gdf, region_name_col="nom_region")
    region_mismatches.to_csv(os.path.join(OUT_DIR, "geosites_region_mismatches.csv"), index=False)

    combined.to_csv(os.path.join(OUT_DIR, "geosites_observations.csv"), index=False)
    print(f"Post-checks: {len(combined)} observations remain", flush=True)

    print("\n=== Step 4: locality construction ===", flush=True)
    localities, needs_review = c01.build_localities(combined)
    localities.to_csv(os.path.join(OUT_DIR, "geosites_localities_master.csv"), index=False)
    needs_review.to_csv(os.path.join(OUT_DIR, "geosites_needs_review.csv"), index=False)

    print(f"\n=== FINAL SUMMARY ===", flush=True)
    print(f"Raw observations parsed: {len(excel_df)}", flush=True)
    print(f"Coordinate parse failures: {len(excel_failures)}", flush=True)
    print(f"Outliers removed (outside Morocco+WS): {len(outliers)}", flush=True)
    print(f"Region mismatches (declared vs geometric): {len(region_mismatches)}", flush=True)
    print(f"Observations after all checks: {len(combined)}", flush=True)
    print(f"Localities constructed: {len(localities)}", flush=True)
    print(f"Median observations/locality: {localities['Observation_Count'].median()}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
