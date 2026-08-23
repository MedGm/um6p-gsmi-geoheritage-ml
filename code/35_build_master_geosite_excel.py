"""
code/35_build_master_geosite_excel.py  (2026-08-18)

Builds the master geosite deliverable requested at the closing meeting: all
1,667 catalogued geosites, in the same descriptive-inventory style as the
original "Data Classification" source spreadsheets (domain, region, name,
type, author/reference, title), but with CORRECTED coordinates (the cleaned,
parsed lat/lon this project's pipeline already produced, not the raw
multi-format strings), plus a new Accessibility_Class column filled in for
the 733 sites with an expert label and left blank for the remaining ~934 --
explicitly so future labeling work has a ready-to-fill template.

Data provenance per column, and the honest coverage gaps (checked, not
assumed):
  - Coordinates, Region: from data/final/geosites_mcdm_national.csv (the
    production catalog), 1,667/1,667.
  - Geological_Domain: from data/newdb_v2/geosites_localities_master.csv
    (original catalog + 2026-08-09 expansion, by Locality_ID) and
    data/newdb_v2_aug16/geosites_localities_vs_final_catalog.csv (2026-08-16
    expansion's new localities, matched by coordinate since those rows were
    renumbered during the final-catalog merge). ~99.8% coverage.
  - Geosite_Type: same two sources. Genuinely sparse in the underlying
    literature sources themselves (~36% coverage) -- not a processing gap.
  - Reference / Title (author + publication citation): recovered from
    data/newdb_v2/labeling_candidates/all_830_with_citations.csv for the
    2026-08-09 batch's 830 new sites (by Locality_ID), and by matching
    Geosite_Name back to the raw 2026-08-16 source file for that batch's 513
    new sites (510/513 matched; 3 failed on name normalization and are left
    blank, not guessed). The ~375 sites from the ORIGINAL pre-2026-08-09
    catalog have NO citation recovered anywhere in this project's processed
    data -- a genuine, pre-existing gap (confirmed by checking, not assumed),
    left blank rather than fabricated. Closing it would require a fresh pass
    against the original source Excel file, not attempted here.
  - Accessibility_Class: from data/final/regional_label_sources/*.csv
    (Expert_Class, "Very Difficult" collapsed into "Difficult" to match every
    other use of this label in the project), 733/1,667. Blank for the rest.

Output: hdbscan/../geosites_master_1667_with_accessibility.xlsx (repo root)
"""
import glob, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT_XLSX = os.path.join(BASE, "geosites_master_1667_with_accessibility.xlsx")

print("Loading production catalog (1,667 sites) ...", flush=True)
final = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
final = final[["Locality_ID", "Geosite_Name", "Region", "Latitude_WGS84", "Longitude_WGS84", "Coordinate_Precision"]].copy()
N = len(final)
assert N == 1667
final["_coord_key"] = list(zip(final["Latitude_WGS84"].round(6), final["Longitude_WGS84"].round(6)))

# ============================================================ Domain / Type
print("Joining Geological_Domain / Geosite_Type ...", flush=True)
master = pd.read_csv(os.path.join(BASE, "data", "newdb_v2", "geosites_localities_master.csv"))
master_dt = master.set_index("Locality_ID")[["Geological_Domain", "Geosite_Type"]]

aug16_dedup = pd.read_csv(os.path.join(BASE, "data", "newdb_v2_aug16", "geosites_localities_vs_final_catalog.csv"))
aug16_new = aug16_dedup[aug16_dedup["match_class"] == "new"].copy()
aug16_new["_coord_key"] = list(zip(aug16_new["Latitude_WGS84"].round(6), aug16_new["Longitude_WGS84"].round(6)))
aug16_dt = aug16_new.drop_duplicates("_coord_key").set_index("_coord_key")[["Geological_Domain", "Geosite_Type"]]

final = final.merge(master_dt, left_on="Locality_ID", right_index=True, how="left")
missing_dt = final["Geological_Domain"].isna()
aug16_lookup = final.loc[missing_dt, "_coord_key"].map(aug16_dt["Geological_Domain"])
final.loc[missing_dt, "Geological_Domain"] = aug16_lookup
aug16_type_lookup = final.loc[missing_dt, "_coord_key"].map(aug16_dt["Geosite_Type"])
final.loc[missing_dt & final["Geosite_Type"].isna(), "Geosite_Type"] = aug16_type_lookup

n_domain = final["Geological_Domain"].notna().sum()
n_type = final["Geosite_Type"].notna().sum()
print(f"  Geological_Domain: {n_domain}/{N} ({100*n_domain/N:.1f}%)", flush=True)
print(f"  Geosite_Type: {n_type}/{N} ({100*n_type/N:.1f}%)", flush=True)

# ============================================================ Reference / Title
print("Joining Reference / Title (author citation) ...", flush=True)
citations = pd.read_csv(os.path.join(BASE, "data", "newdb_v2", "labeling_candidates", "all_830_with_citations.csv"))
cit_lookup = citations.set_index("Locality_ID")[["Reference", "Title"]]
final = final.merge(cit_lookup, left_on="Locality_ID", right_index=True, how="left")

raw_aug16 = pd.read_csv(
    os.path.join(BASE, "references", "databases", "new-db-aug16", "Data Classification_16-08-2026(11-08-2026).csv"),
    skiprows=1, encoding="utf-8-sig")
raw_aug16.columns = ["Geological_domain", "Admin_region", "Auteurs", "Titre", "Geosite_name", "Geosite_type",
                      "X_Easting", "Y_Northing"] + [f"extra_{i}" for i in range(len(raw_aug16.columns) - 8)]
raw_aug16["Auteurs"] = raw_aug16["Auteurs"].ffill()
raw_aug16["Titre"] = raw_aug16["Titre"].ffill()
name_to_cite = raw_aug16.dropna(subset=["Geosite_name"]).drop_duplicates("Geosite_name", keep="first") \
    .set_index("Geosite_name")[["Auteurs", "Titre"]]

new_locs = pd.read_csv(os.path.join(BASE, "data", "newdb_v2_aug16", "geosites_new_localities_features.csv"))
aug16_new_coordkey = pd.DataFrame({
    "_coord_key": list(zip(new_locs["Latitude_WGS84"].round(6), new_locs["Longitude_WGS84"].round(6))),
    "Geosite_Name": new_locs["Geosite_Name"].values,
}).drop_duplicates("_coord_key").set_index("_coord_key")

missing_cite = final["Reference"].isna()
name_for_missing = final.loc[missing_cite, "_coord_key"].map(aug16_new_coordkey["Geosite_Name"])
ref_for_missing = name_for_missing.map(name_to_cite["Auteurs"])
title_for_missing = name_for_missing.map(name_to_cite["Titre"])
final.loc[missing_cite, "Reference"] = ref_for_missing
final.loc[missing_cite, "Title"] = title_for_missing

n_cite = final["Reference"].notna().sum()
print(f"  Reference/Title: {n_cite}/{N} ({100*n_cite/N:.1f}%) -- remaining {N-n_cite} are the original "
      f"pre-2026-08-09 catalog (no citation recovered anywhere in processed data) plus a handful of "
      f"2026-08-16 name-matching misses", flush=True)

# ============================================================ Accessibility class
print("Joining Accessibility_Class (733 expert-labeled sites) ...", flush=True)
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
all_labels["Accessibility_Class"] = all_labels["Expert_Class"].replace("Very Difficult", "Difficult")
# 734 unique labeled IDs exist, but one ("Dayet Aoua", loc_00717) has a null Region
# in the production catalog -- the same gap that makes every other script in this
# project (code/13, 20, 24, ...) apply dropna(subset=["Region"]) and land on 733,
# not 734. Matched here for consistency with every other "N=733" figure already
# published, rather than silently shipping a different count in this one file.
region_lookup = final.set_index("Locality_ID")["Region"]
all_labels = all_labels[all_labels["Locality_ID"].map(region_lookup).notna()]
final = final.merge(all_labels[["Locality_ID", "Accessibility_Class"]], on="Locality_ID", how="left")
n_labeled = final["Accessibility_Class"].notna().sum()
print(f"  Accessibility_Class: {n_labeled}/{N} ({100*n_labeled/N:.1f}%)", flush=True)
assert n_labeled >= 733, f"expected at least the original 733 labels, got {n_labeled}"

# ============================================================ Assemble + save
final = final.drop(columns=["_coord_key"])
final = final.rename(columns={
    "Latitude_WGS84": "Latitude_WGS84_corrected",
    "Longitude_WGS84": "Longitude_WGS84_corrected",
})
out_cols = ["Locality_ID", "Geological_Domain", "Region", "Geosite_Name", "Geosite_Type",
            "Reference", "Title", "Latitude_WGS84_corrected", "Longitude_WGS84_corrected",
            "Coordinate_Precision", "Accessibility_Class"]
final = final[out_cols].sort_values("Locality_ID").reset_index(drop=True)

final.to_excel(OUT_XLSX, index=False, sheet_name="Geosites_1667")
print(f"\nSaved {OUT_XLSX} ({len(final)} rows)", flush=True)
print("\nColumn coverage summary:")
for c in out_cols:
    n = final[c].notna().sum()
    print(f"  {c}: {n}/{N} ({100*n/N:.1f}%)")
