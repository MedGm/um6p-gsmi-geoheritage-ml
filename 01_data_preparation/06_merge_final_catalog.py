"""
code/33_merge_aug16_into_final_catalog.py  (2026-08-17)

Appends the featurized "new" localities from code/32 (the 2026-08-16
supervisor-provided source, after code/31's parsing/border/region/dedup
checks) to the production catalog (data/final/geosites_mcdm_national.csv),
following the exact same documented policy as the Aug-9 expansion: NO
accessibility label is assigned (MCDM_Score/MCDM_Class/National_Model_Class
left blank) -- these sites have no expert label, and this project has a
hard-won, repeatedly-documented lesson against formula-derived/circular
labels. They ARE usable immediately for the geosite-location favorability
model (code/13-style retrain), which only needs "is this a real geosite
location" -- these sites qualify, same reasoning as code/12's docstring.

Locality_ID collision note: code/01's build_localities() always starts
counting from loc_00001 within whatever source it's given, so the Aug16
batch's own locality IDs (from code/31's output) collide with the existing
1-1205 numbering in the production catalog. Renumbered here to continue from
the current max (loc_01206 onward), not reused verbatim.

Writes a backup of the pre-merge catalog before overwriting, matching this
project's established practice of never overwriting a production file
without a recoverable copy.
"""
import os
import shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
NEWDB = os.path.join(BASE, "data", "newdb_v2_aug16")
FINAL_CSV = os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv")

FEATURES = ["Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Highway_m", "LULC_Friction", "Dist_to_Settlement_m"]


def main():
    final = pd.read_csv(FINAL_CSV)
    new = pd.read_csv(os.path.join(NEWDB, "geosites_new_localities_features.csv"))
    print(f"Current catalog: {len(final)} sites", flush=True)
    print(f"New localities to merge: {len(new)}", flush=True)

    n_before_feature_drop = len(new)
    new = new.dropna(subset=FEATURES + ["Latitude_WGS84", "Longitude_WGS84"])
    if len(new) < n_before_feature_drop:
        print(f"  dropped {n_before_feature_drop - len(new)} rows with missing core features "
              f"after median-fill safeguard failed (should be rare/zero)", flush=True)

    existing_ids = final["Locality_ID"].str.extract(r"(\d+)")[0].astype(int)
    next_id = existing_ids.max() + 1
    new = new.reset_index(drop=True)
    new["Locality_ID"] = [f"loc_{next_id + i:05d}" for i in range(len(new))]
    print(f"Renumbered new localities loc_{next_id:05d} .. loc_{next_id + len(new) - 1:05d}", flush=True)

    for col in ["LULC_Class_Name_WorldCover", "N_Raster_Cells_Imputed", "MCDM_Score", "MCDM_Class", "National_Model_Class"]:
        if col not in new.columns:
            new[col] = np.nan
    # Every existing row in the production catalog is "point" precision (verified); code/31's
    # parser only ever returns "point" or "interval" per observation, and build_localities()
    # doesn't carry that column through to its locality-level output, so default to the
    # overwhelmingly-dominant existing convention rather than leaving it blank.
    if "Coordinate_Precision" not in new.columns:
        new["Coordinate_Precision"] = "point"

    new = new[final.columns.tolist()]

    backup_path = FINAL_CSV.replace(".csv", f"_pre_aug16_merge_backup.csv")
    shutil.copy(FINAL_CSV, backup_path)
    print(f"Backed up pre-merge catalog to {backup_path}", flush=True)

    merged = pd.concat([final, new], ignore_index=True)
    assert merged["Locality_ID"].is_unique, "Locality_ID collision after merge -- aborting before write"
    merged.to_csv(FINAL_CSV, index=False)
    print(f"\nMerged catalog written: {len(merged)} sites total ({len(final)} + {len(new)} new)", flush=True)
    print(f"  Geology_Class missing: {merged['Geology_Class'].isna().sum()} "
          f"({new['Geology_Class'].isna().sum()} from new batch)", flush=True)
    print(f"  Soil_Class missing: {merged['Soil_Class'].isna().sum()} "
          f"({new['Soil_Class'].isna().sum()} from new batch)", flush=True)
    print(f"  Labeled (MCDM_Class non-null): {merged['MCDM_Class'].notna().sum()} "
          f"(unchanged -- new sites carry no accessibility label, by design)", flush=True)


if __name__ == "__main__":
    main()
