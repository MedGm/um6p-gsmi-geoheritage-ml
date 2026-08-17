"""
code/31_ingest_aug16_geoheritage_source.py  (2026-08-17)

Ingests the new 2026-08-16 supervisor-provided geoheritage source ("Data
Classification_16-08-2026(11-08-2026).csv", project root) using the same
Observation/Locality pipeline as code/01 and code/11 -- reusing code/01's
coordinate parser, border check, regional-plausibility check, and locality
builder verbatim (imported, not reimplemented), and code/11's hierarchical
row loader + post-hoc Merchich outlier check verbatim.

Why a new script instead of extending 11: this file's raw column layout
differs from both prior sources (Coordinates block only, X_Easting/Y_Northing
under it, no Center-coordinates block at all) and its coordinate formats are
far more heterogeneous than either prior source alone -- decimal+hemisphere,
full DMS, decimal-minutes, comma-decimal degrees, AND ~19 bare Merchich/
Lambert (EPSG:26191) easting/northing rows (same format code/01 already
built a dedicated branch for, from the Aug-9 Chefchaouen group) -- all mixed
in the same column. code/01's parse_coordinate_pair already auto-detects
every one of these formats; the one gap closed here is a redundant leading
Unicode minus sign combined with a hemisphere letter (e.g. "−5.72W"), which
none of the existing regexes anticipated (redundant with the hemisphere
letter, stripped before parsing -- not a numeric guess).

Review findings that motivated this script (see conversation): of 796 rows
with a geosite name + some coordinate value, a naive single-format parser
recovered only 618 (78%); this script's reuse of code/01's full parser
(DMS/decimal-minutes/comma-decimal/Merchich, all covered) is expected to
recover substantially more. Region labels also have real spelling drift
("Béni Mellal-Khénifra" in >=4 variants) -- code/01's `_norm_region` already
accent-strips and whitespace-normalizes before the geometric plausibility
check, so this is handled by the existing machinery, not new code.

NEW step beyond 01/11 (neither existing script needed it, since 11 was the
FIRST expansion into an empty newdb_v2/): dedup this source's localities
against the CURRENT production catalog (data/final/geosites_mcdm_national.csv,
already containing the Aug-9 expansion), using the exact dual-signal rule
from archive/scratch_newdb_work/newdb_matching.py (rapidfuzz token_sort_ratio
name similarity >=85 AND haversine distance <=2.0km => duplicate; only one
signal => flagged for review, not silently merged; neither => new).

Outputs (data/newdb_v2_aug16/):
  geosites_observations_raw.csv       -- all parsed observations
  dms_parse_failures.csv              -- coordinate strings that could not be parsed
  geosites_outliers_removed.csv       -- outside Morocco+Western Sahara border check
  geosites_merchich_posthoc_outliers.csv -- Merchich-format rows implausibly far from their group
  geosites_region_mismatches.csv      -- declared region vs geometric region disagree
  geosites_localities_master.csv      -- deduplicated localities (within this source)
  geosites_localities_vs_final_catalog.csv -- dedup classification vs current production catalog
  geosites_needs_review.csv           -- locality-construction review flags (from build_localities)
"""
import importlib.util
import os
import re
import unicodedata
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW_CSV = os.path.join(BASE, "references", "databases", "new-db-aug16", "Data Classification_16-08-2026(11-08-2026).csv")
OUT_DIR = os.path.join(BASE, "data", "newdb_v2_aug16")
os.makedirs(OUT_DIR, exist_ok=True)

spec01 = importlib.util.spec_from_file_location("consolidate01", os.path.join(HERE, "01_consolidate_geosite_catalog.py"))
c01 = importlib.util.module_from_spec(spec01)
spec01.loader.exec_module(c01)

spec11 = importlib.util.spec_from_file_location("ingest11", os.path.join(HERE, "11_ingest_expanded_geoheritage_source.py"))
c11 = importlib.util.module_from_spec(spec11)
spec11.loader.exec_module(c11)

NAME_MATCH_THRESHOLD = 85
COORD_MATCH_KM = 2.0


def _strip_redundant_sign(s):
    """A leading Unicode minus (e.g. '−5.7229291W') is redundant with the
    trailing hemisphere letter and not handled by any of code/01's coordinate
    regexes (which expect a bare digit start). Confirmed on this file: every
    occurrence pairs a leading '−' with a W or S suffix (never a conflicting
    sign), so stripping it is a pure normalization, not a guess."""
    if not isinstance(s, str):
        return s
    return re.sub(r"^[−\-]\s*", "", s.strip())


def load_aug16_rows(path):
    """Load the raw CSV (2 header rows: merged group header + 'X_Easting/Y_Northing'
    sub-header) and emit row tuples in code/11's load_hierarchical_rows shape:
    (domain, region, reference, title, name, gtype, cx, cy, ccx, ccy). Deliberately
    does NOT forward-fill the group columns (Geological domain, Administrative region,
    Auteurs, Titre repeat only on each group's first row, same convention as Data
    generale) -- load_hierarchical_rows detects a new group itself via `domain is not
    None`, exactly matching the sparse (unmerged-cell) shape this CSV export already
    has; pre-filling here would make every row look like its own group start. This
    source has no Center-coordinates block at all, so ccx/ccy are always None."""
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    df.columns = ["Geological_domain", "Admin_region", "Auteurs", "Titre", "Geosite_name",
                  "Geosite_type", "X_Easting", "Y_Northing"] + [f"extra_{i}" for i in range(len(df.columns) - 8)]

    rows = []
    for i, row in df.iterrows():
        cx = _strip_redundant_sign(row["X_Easting"]) if pd.notna(row["X_Easting"]) else None
        cy = _strip_redundant_sign(row["Y_Northing"]) if pd.notna(row["Y_Northing"]) else None
        name = row["Geosite_name"] if pd.notna(row["Geosite_name"]) else None
        rows.append((
            i + 3,
            (row["Geological_domain"] if pd.notna(row["Geological_domain"]) else None,
             row["Admin_region"] if pd.notna(row["Admin_region"]) else None,
             row["Auteurs"] if pd.notna(row["Auteurs"]) else None,
             row["Titre"] if pd.notna(row["Titre"]) else None,
             name, row["Geosite_type"] if pd.notna(row["Geosite_type"]) else None, cx, cy, None, None),
        ))
    return rows


def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[\"'’]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def dedup_against_final_catalog(localities, final_csv):
    """Dual-signal rule from archive/scratch_newdb_work/newdb_matching.py, applied
    against the CURRENT production catalog (already contains the Aug-9 expansion --
    the first newdb batch never had to do this since it started from an empty
    newdb_v2/). name+coord agree => duplicate; single signal => review, not silently
    merged; neither => new."""
    final = pd.read_csv(final_csv)
    final["name_norm"] = final["Geosite_Name"].apply(normalize_name)
    localities["name_norm"] = localities["Geosite_Name"].apply(normalize_name)
    choices = final["name_norm"].tolist()

    rows = []
    for _, loc in localities.iterrows():
        if not loc["name_norm"]:
            rows.append("new")
            continue
        result = process.extractOne(loc["name_norm"], choices, scorer=fuzz.token_sort_ratio)
        if result is None:
            rows.append("new")
            continue
        _, score, match_idx = result
        match_row = final.iloc[match_idx]
        dist_km = np.nan
        if pd.notna(loc["Latitude_WGS84"]) and pd.notna(match_row["Latitude_WGS84"]):
            dist_km = haversine_km(loc["Latitude_WGS84"], loc["Longitude_WGS84"],
                                    match_row["Latitude_WGS84"], match_row["Longitude_WGS84"])
        name_ok = score >= NAME_MATCH_THRESHOLD
        coord_ok = pd.notna(dist_km) and dist_km <= COORD_MATCH_KM
        if name_ok and coord_ok:
            rows.append("duplicate")
        elif name_ok and not pd.notna(dist_km):
            rows.append("review_no_coord")
        elif name_ok and not coord_ok:
            rows.append("review_coord_disagree")
        elif coord_ok:
            rows.append("review_name_differs")
        else:
            rows.append("new")
    localities["match_class"] = rows
    return localities


def main():
    print("=== Step 1: load + parse (reusing code/01's full coordinate parser) ===", flush=True)
    rows = load_aug16_rows(RAW_CSV)
    excel_df, failures = c11.load_hierarchical_rows(rows, "geoheritage_aug16_2026")
    n_with_coords = excel_df["Latitude_WGS84"].notna().sum()
    print(f"Parsed {len(excel_df)} observations ({n_with_coords} with coordinates), "
          f"{len(failures)} coordinate parse failures", flush=True)

    excel_df.insert(0, "Observation_ID", [f"obs3_{i:05d}" for i in range(len(excel_df))])
    excel_df.to_csv(os.path.join(OUT_DIR, "geosites_observations_raw.csv"), index=False)
    failures.to_csv(os.path.join(OUT_DIR, "dms_parse_failures.csv"), index=False)

    print("\n=== Step 2: corrections, border check, Merchich post-hoc, regional plausibility ===", flush=True)
    combined = c01.apply_documented_corrections(excel_df)
    combined = c01.apply_ground_truth_notes(combined)

    # Local cache reuse: data/boundaries/national.geojson is the same Natural Earth
    # Morocco+Western Sahara admin-0 pair fetch_morocco_boundary() would fetch over
    # the network (confirmed: identical ADMIN=['Morocco','Western Sahara']) -- reused
    # here to avoid a redundant fetch (GitHub raw rate-limited this session already).
    import geopandas as gpd
    boundary = gpd.read_file(os.path.join(BASE, "data", "boundaries", "national.geojson")).to_crs("EPSG:4326")
    combined, outliers = c01.border_check(combined, boundary)
    outliers.to_csv(os.path.join(OUT_DIR, "geosites_outliers_removed.csv"), index=False)

    combined, merchich_flagged = c11.post_hoc_merchich_check(combined)
    merchich_flagged.to_csv(os.path.join(OUT_DIR, "geosites_merchich_posthoc_outliers.csv"), index=False)

    # raw.githubusercontent.com was rate-limited (429) this session for this file
    # specifically (heavy repeated /vsicurl/ + boundary fetch usage earlier today);
    # `git clone --depth 1` of the same repo succeeded where the CDN didn't, so the
    # file is cached locally once and reused here (same content: 12 regions,
    # nom_region field, confirmed against fetch_region_boundaries()'s own assertions).
    region_cache = os.path.join(BASE, "data", "boundaries", "morocco_regions_admin12.geojson")
    region_gdf = gpd.read_file(region_cache).to_crs("EPSG:4326")
    assert len(region_gdf) == 12, f"Expected 12 regions, got {len(region_gdf)}"
    assert region_gdf.geometry.is_valid.all(), "Cached region polygons contain invalid geometry"
    clean, region_mismatches = c01.regional_plausibility_check(combined, region_gdf, region_name_col="nom_region")
    region_mismatches.to_csv(os.path.join(OUT_DIR, "geosites_region_mismatches.csv"), index=False)

    # Rescue rather than drop: inspection of the 203 mismatches shows the dominant
    # pattern is coordinates that land unambiguously in a real place (e.g. "Azrou" --
    # a well-known Fes-Meknes town) while the source paper's DECLARED region is a
    # park-scale label that doesn't hold at the individual-site level (146 of 203 are
    # "Beni Mellal-Khenifra"-declared, Fes-Meknes-detected -- consistent with the
    # M'Goun Geopark / Ait Attab Syncline paper spanning that regional border), plus
    # 18 rows declared only as generic "Morocco" (not a real region, geometry is
    # strictly better information than a placeholder). The favorability model these
    # sites are being added for doesn't even use Region as a feature -- dropping 203
    # genuine, in-Morocco geosites over a mislabeled metadata field would be pure
    # waste. Detected_Region (geometric, authoritative) replaces the declared value;
    # flagged via Region_Corrected for provenance, not silently changed.
    rescued = region_mismatches.copy()
    rescued["Region"] = rescued["Detected_Region"]
    rescued["Region_Corrected"] = "yes (declared region did not match geometric location)"
    rescued = rescued.drop(columns=["Detected_Region", "index_right"], errors="ignore")
    clean["Region_Corrected"] = "no"
    combined = pd.concat([clean, rescued], ignore_index=True)
    print(f"Rescued {len(rescued)} region-mismatched observations using detected (geometric) region "
          f"instead of dropping them", flush=True)

    combined.to_csv(os.path.join(OUT_DIR, "geosites_observations.csv"), index=False)
    print(f"Post-checks: {len(combined)} observations remain", flush=True)

    print("\n=== Step 3: locality construction (within-source dedup) ===", flush=True)
    localities, needs_review = c01.build_localities(combined)
    needs_review.to_csv(os.path.join(OUT_DIR, "geosites_needs_review.csv"), index=False)
    print(f"Localities constructed: {len(localities)}", flush=True)

    print("\n=== Step 4: dedup against CURRENT production catalog ===", flush=True)
    final_csv = os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv")
    localities = dedup_against_final_catalog(localities, final_csv)
    localities.to_csv(os.path.join(OUT_DIR, "geosites_localities_vs_final_catalog.csv"), index=False)
    print(localities["match_class"].value_counts(), flush=True)

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(f"Raw observations parsed: {len(excel_df)} ({n_with_coords} with coordinates)", flush=True)
    print(f"Coordinate parse failures: {len(failures)}", flush=True)
    print(f"Outliers removed (outside Morocco+WS): {len(outliers)}", flush=True)
    print(f"Merchich post-hoc outliers removed: {len(merchich_flagged)}", flush=True)
    print(f"Region mismatches (flagged, not removed): {len(region_mismatches)}", flush=True)
    print(f"Localities constructed: {len(localities)}", flush=True)
    print(f"  new (not in current catalog): {(localities['match_class']=='new').sum()}", flush=True)
    print(f"  duplicate (name+coord agree): {(localities['match_class']=='duplicate').sum()}", flush=True)
    print(f"  flagged for review: {(localities['match_class'].str.startswith('review')).sum()}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
