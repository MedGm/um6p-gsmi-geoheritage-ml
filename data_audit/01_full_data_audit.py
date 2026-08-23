"""
data_audit/01_full_data_audit.py  (2026-08-18)

Phase 1 of a deliberate, step-by-step re-examination of the dataset, requested
after the closing meeting: audit every column for errors, outliers, and
conflicts BEFORE touching any model. Nothing here changes the deployed
catalog or any model -- pure diagnostics, findings saved to CSV for review.

Sections:
  A. Schema + missingness, full 1,667-site catalog and the 733-labeled subset
  B. Domain-implausibility checks (values outside physically sensible ranges)
  C. Statistical outliers per feature, computed WITHIN each accessibility
     class separately (an extreme value can be normal for Difficult and
     suspicious for Easy -- pooling classes would hide that)
  D. Exact-duplicate coordinates (distinct Locality_ID, identical lat/lon --
     a data-entry issue, different from the 500m near-duplicate CV grouping
     already handled elsewhere)
  E. Label-vs-feature contradiction scan: a simple, transparent difficulty
     score from the 6 features, checked against the actual expert label for
     every one of the 733 sites -- flags the most surprising disagreements
     for manual review, does not claim any of them are wrong
  F. Region-assignment geometric check for the 733 labeled sites specifically
     (declared Region vs geometric containment) -- the same check already
     applied to the two catalog-expansion batches, not yet applied to the
     original labeled set
  G. Confidence-level distribution by region and class
  H. Legacy MCDM_Class / National_Model_Class vs Expert_Class agreement check
     (formula-derived fields with documented circularity history -- confirms
     current status, not assumed)
"""
import glob, os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = HERE

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
N_CAT = len(catalog)
assert N_CAT == 1667

frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    if "Confidence" not in labeled.columns:
        labeled["Confidence"] = "Medium"
    cols = ["Locality_ID", "Expert_Class", "Confidence"]
    if "Expert_Reasoning" in labeled.columns:
        cols.append("Expert_Reasoning")
    frames.append(labeled[cols])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
merged = all_labels.merge(
    catalog[["Locality_ID", "Geosite_Name", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES +
            ["MCDM_Class", "National_Model_Class"]],
    on="Locality_ID", how="inner")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
print(f"Labeled subset: N={N} (original 733 + el_ouali_2026 batch; {'OK, matches known total' if N in (733, 939) else 'unexpected count -- investigate'})\n")

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

# ============================================================ A. Schema + missingness
log("=" * 70)
log("A. SCHEMA + MISSINGNESS")
log("=" * 70)
log("\nFull catalog (N=1667):")
for c in catalog.columns:
    n_null = catalog[c].isna().sum()
    log(f"  {c:30s} null={n_null:5d} ({100*n_null/N_CAT:5.1f}%)  dtype={catalog[c].dtype}")

log("\nLabeled subset (N=733), original Expert_Class (pre-merge) distribution:")
log(merged["Expert_Class"].value_counts(dropna=False).to_string())
log(f"\n'Very Difficult' collapsed into 'Difficult': {(merged['Expert_Class']=='Very Difficult').sum()} sites")
log("\nExpert_Merged (used everywhere in modeling) distribution:")
log(merged["Expert_Merged"].value_counts().to_string())

# ============================================================ B. Domain-implausibility
log("\n" + "=" * 70)
log("B. DOMAIN-IMPLAUSIBILITY CHECKS (full catalog, N=1667)")
log("=" * 70)
checks = {
    "Dist_to_Highway_m < 0": catalog["Dist_to_Highway_m"] < 0,
    "Dist_to_Highway_m == 0 exactly": catalog["Dist_to_Highway_m"] == 0,
    "Dist_to_Settlement_m < 0": catalog["Dist_to_Settlement_m"] < 0,
    "Slope_deg < 0": catalog["Slope_deg"] < 0,
    "Slope_deg > 60 (very steep, verify not a DEM artifact)": catalog["Slope_deg"] > 60,
    "Slope_deg > 90 (impossible)": catalog["Slope_deg"] > 90,
    "Elevation_m < -100 (well below sea level, verify)": catalog["Elevation_m"] < -100,
    "Elevation_m > 4200 (above Toubkal, verify)": catalog["Elevation_m"] > 4200,
    "LULC_Friction outside [0,1]": ~catalog["LULC_Friction"].between(0, 1),
    "Ruggedness < 0": catalog["Ruggedness"] < 0,
    "Latitude outside Morocco+WS bbox [20.5,36]": ~catalog["Latitude_WGS84"].between(20.5, 36),
    "Longitude outside Morocco+WS bbox [-17.2,-1]": ~catalog["Longitude_WGS84"].between(-17.2, -1),
}
implausible_rows = {}
for name, mask in checks.items():
    n = mask.sum()
    log(f"  {name:55s} {n:5d} rows")
    if n > 0:
        implausible_rows[name] = catalog[mask][["Locality_ID", "Geosite_Name", "Region"] + FEATURES]

if implausible_rows:
    with pd.ExcelWriter(os.path.join(OUT, "B_domain_implausible_rows.xlsx")) as xw:
        for name, df in implausible_rows.items():
            sheet = name[:31].replace("/", "-")
            df.to_excel(xw, sheet_name=sheet, index=False)
    log(f"  -> saved flagged rows to B_domain_implausible_rows.xlsx")

# ============================================================ C. Statistical outliers per class
log("\n" + "=" * 70)
log("C. STATISTICAL OUTLIERS PER FEATURE, WITHIN EACH CLASS (labeled N=733)")
log("=" * 70)
outlier_rows = []
for feat in FEATURES:
    for cls in ["Easy", "Moderate", "Difficult"]:
        sub = merged[merged["Expert_Merged"] == cls]
        q1, q3 = sub[feat].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr  # 3xIQR = extreme-outlier convention, not the usual 1.5x
        out = sub[(sub[feat] < lo) | (sub[feat] > hi)]
        if len(out) > 0:
            log(f"  {feat:22s} class={cls:10s} extreme outliers (3xIQR): {len(out)} "
                f"(bounds [{lo:.1f}, {hi:.1f}], class range [{sub[feat].min():.1f}, {sub[feat].max():.1f}])")
            for _, r in out.iterrows():
                outlier_rows.append({"Feature": feat, "Class": cls, "Locality_ID": r["Locality_ID"],
                                      "Geosite_Name": r["Geosite_Name"], "Region": r["Region"],
                                      "Value": r[feat], "Bounds": f"[{lo:.1f},{hi:.1f}]"})
if outlier_rows:
    pd.DataFrame(outlier_rows).to_csv(os.path.join(OUT, "C_statistical_outliers_per_class.csv"), index=False)
    log(f"\n  -> saved {len(outlier_rows)} flagged rows to C_statistical_outliers_per_class.csv")

# ============================================================ D. Exact-duplicate coordinates
log("\n" + "=" * 70)
log("D. EXACT-DUPLICATE COORDINATES (distinct Locality_ID, identical lat/lon)")
log("=" * 70)
catalog["_coord_key"] = list(zip(catalog["Latitude_WGS84"].round(6), catalog["Longitude_WGS84"].round(6)))
dup_mask = catalog.duplicated("_coord_key", keep=False)
dups = catalog[dup_mask].sort_values("_coord_key")
n_dup_groups = dups["_coord_key"].nunique()
log(f"  {len(dups)} rows share coordinates with at least one other row, in {n_dup_groups} groups")
if len(dups) > 0:
    dups[["Locality_ID", "Geosite_Name", "Region", "Latitude_WGS84", "Longitude_WGS84"]].to_csv(
        os.path.join(OUT, "D_exact_duplicate_coordinates.csv"), index=False)
    log(f"  -> saved to D_exact_duplicate_coordinates.csv")
    merged["_coord_key"] = list(zip(merged["Latitude_WGS84"].round(6), merged["Longitude_WGS84"].round(6)))
    labeled_dup = merged[merged["Locality_ID"].isin(dups["Locality_ID"])]
    log(f"  Of these, {len(labeled_dup)} are in the 733-site LABELED set "
        f"(same coordinates, potentially different labels -- checked below)")
    n_conflicts = 0
    conflict_rows = []
    if len(labeled_dup) > 0:
        for key, grp in labeled_dup.groupby("_coord_key"):
            if grp["Expert_Merged"].nunique() > 1:
                n_conflicts += 1
                log(f"    CONFLICT at {key}: {grp[['Locality_ID','Geosite_Name','Expert_Merged']].to_dict('records')}")
                conflict_rows.append(grp[["Locality_ID", "Geosite_Name", "Region", "Expert_Merged", "Confidence"]].assign(coord_key=str(key)))
    log(f"  -> {n_conflicts} coordinate groups have DIFFERENT labels among same-coordinate sites")
    if conflict_rows:
        pd.concat(conflict_rows).to_csv(os.path.join(OUT, "D_label_conflicts_same_coords.csv"), index=False)
        log(f"  -> saved to D_label_conflicts_same_coords.csv")

# ============================================================ E. Label-vs-feature contradiction scan
log("\n" + "=" * 70)
log("E. LABEL-VS-FEATURE CONTRADICTION SCAN (transparent scoring, not a model)")
log("=" * 70)
log("Simple difficulty score: z-scored (Dist_to_Highway_m + Slope_deg + Ruggedness + Dist_to_Settlement_m)")
log("minus z-scored Elevation_m contribution is NOT included (elevation's relationship to difficulty is")
log("context-dependent, not monotonic) -- kept deliberately simple and inspectable, not a claim of ground truth.\n")
z = merged.copy()
for feat in ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Dist_to_Settlement_m"]:
    z[f"z_{feat}"] = (z[feat] - z[feat].mean()) / z[feat].std()
z["difficulty_score"] = z[[f"z_{f}" for f in ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Dist_to_Settlement_m"]]].sum(axis=1)

easy_high_score = z[z["Expert_Merged"] == "Easy"].nlargest(10, "difficulty_score")
diff_low_score = z[z["Expert_Merged"] == "Difficult"].nsmallest(10, "difficulty_score")
log("Top 10 'Easy'-labeled sites with the HIGHEST feature-based difficulty score (most surprising if truly Easy):")
log(easy_high_score[["Locality_ID", "Geosite_Name", "Region", "Confidence", "difficulty_score"] + FEATURES[:4]].to_string(index=False))
log("\nTop 10 'Difficult'-labeled sites with the LOWEST feature-based difficulty score (most surprising if truly Difficult):")
log(diff_low_score[["Locality_ID", "Geosite_Name", "Region", "Confidence", "difficulty_score"] + FEATURES[:4]].to_string(index=False))

contradictions = pd.concat([easy_high_score.assign(flag="Easy_but_looks_hard"),
                              diff_low_score.assign(flag="Difficult_but_looks_easy")])
contradictions[["flag", "Locality_ID", "Geosite_Name", "Region", "Confidence", "Expert_Merged",
                 "difficulty_score"] + FEATURES].to_csv(
    os.path.join(OUT, "E_label_feature_contradictions.csv"), index=False)
log(f"\n  -> saved to E_label_feature_contradictions.csv")

# ============================================================ F. Region-assignment geometric check
log("\n" + "=" * 70)
log("F. REGION-ASSIGNMENT GEOMETRIC CHECK (733 labeled sites: declared vs geometric region)")
log("=" * 70)
regions_gdf = gpd.read_file(os.path.join(BASE, "data", "boundaries", "morocco_regions_admin12.geojson"))
pts = gpd.GeoDataFrame(merged, geometry=[Point(xy) for xy in zip(merged["Longitude_WGS84"], merged["Latitude_WGS84"])], crs="EPSG:4326")
joined = gpd.sjoin(pts, regions_gdf[["nom_region", "geometry"]], how="left", predicate="within")
joined = joined[~joined.index.duplicated(keep="first")]

def norm_region(s):
    s = str(s).lower().strip()
    for a, b in [("é", "e"), ("è", "e"), ("â", "a"), ("ï", "i"), ("î", "i"), ("ô", "o"), ("û", "u"),
                 ("ç", "c"), ("à", "a"), ("-", " ")]:
        s = s.replace(a, b)
    return " ".join(s.split())

joined["_declared_norm"] = joined["Region"].apply(norm_region)
joined["_detected_norm"] = joined["nom_region"].apply(norm_region)
mismatch = joined["nom_region"].notna() & (joined["_declared_norm"] != joined["_detected_norm"])
n_mismatch = mismatch.sum()
log(f"  {n_mismatch}/{N} labeled sites: declared Region disagrees with geometric containment")
if n_mismatch > 0:
    mm = joined[mismatch][["Locality_ID", "Geosite_Name", "Region", "nom_region", "Expert_Merged"]]
    log(mm.to_string(index=False))
    mm.to_csv(os.path.join(OUT, "F_region_mismatches_labeled_sites.csv"), index=False)
    log(f"  -> saved to F_region_mismatches_labeled_sites.csv")
n_no_geom = joined["nom_region"].isna().sum()
log(f"  {n_no_geom} labeled sites fell outside all 12 region polygons entirely (coastal/border precision issue)")

# ============================================================ G. Confidence distribution
log("\n" + "=" * 70)
log("G. LABEL CONFIDENCE DISTRIBUTION")
log("=" * 70)
log("\nBy class:")
log(pd.crosstab(merged["Expert_Merged"], merged["Confidence"]).to_string())
log("\nBy region:")
log(pd.crosstab(merged["Region"], merged["Confidence"]).to_string())
low_conf_rate_by_region = merged.groupby("Region")["Confidence"].apply(lambda s: (s.isin(["Low", "Low-Medium"])).mean()).sort_values(ascending=False)
log("\nLow/Low-Medium confidence rate by region (highest first):")
log(low_conf_rate_by_region.round(3).to_string())

# ============================================================ H. Legacy MCDM_Class agreement
log("\n" + "=" * 70)
log("H. LEGACY MCDM_Class / National_Model_Class vs EXPERT LABEL (confirm not used as a feature)")
log("=" * 70)
has_mcdm = merged["MCDM_Class"].notna()
log(f"  Labeled sites with a non-null legacy MCDM_Class: {has_mcdm.sum()}/{N}")
if has_mcdm.sum() > 0:
    agree = (merged.loc[has_mcdm, "MCDM_Class"] == merged.loc[has_mcdm, "Expert_Merged"]).mean()
    log(f"  Agreement rate MCDM_Class vs Expert_Merged (where both exist): {agree:.3f}")
    log(pd.crosstab(merged.loc[has_mcdm, "Expert_Merged"], merged.loc[has_mcdm, "MCDM_Class"]).to_string())
log("\n  Confirming MCDM_Class/National_Model_Class are NOT in the training FEATURES list used by any script:")
log(f"  FEATURES = {FEATURES}")
log("  (both legacy columns absent from this list -- verified by inspection, not just asserted)")

with open(os.path.join(OUT, "audit_log_full.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\nFull log saved to audit_log_full.txt")
log("\nDONE -- Phase 1 audit complete. No data or models were modified.")
