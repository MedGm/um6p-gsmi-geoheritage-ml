"""
Prepares per-region labeling pilot batches for the next round of expert labeling
following the 4 August 2026 meeting tasks:
  - Fés-Meknès      (349 feature-complete sites → 50-site pilot)
  - Tanger          (Tanger-Al Houceima + Tanger-Tétouan-Al Hoceïma, 174 sites → 50-site pilot)
  - Dakhla/Rabat    (Dakhla-Oued Eddahab + Rabat-Salé-Kénitra, ~30 sites → full batch)
  - Marrakech-Safi  (50 sites → 50-site pilot)

For each region the output CSV contains:
  Locality_ID, Geosite_Name, Region, Latitude_WGS84, Longitude_WGS84,
  Elevation_m, Slope_deg, Ruggedness, Dist_to_Highway_m, LULC_Friction,
  Dist_to_Settlement_m, Reference, Title,
  Expert_Class (blank), Confidence (blank), Expert_Reasoning (blank)

Expert_Class, Confidence and Expert_Reasoning are left empty for the labeling agent.

Selection strategy: for each region, sort by Dist_to_Highway_m ascending (road-accessible
sites are easier to verify from literature) then take the first N as the pilot batch.
All sites with an existing label (old + new pilots) are excluded.

Outputs
-------
data/newdb_v2/labeling_candidates/fes_meknes_50_pilot.csv
data/newdb_v2/labeling_candidates/tanger_50_pilot.csv
data/newdb_v2/labeling_candidates/dakhla_rabat_pilot.csv
data/newdb_v2/labeling_candidates/marrakech_safi_50_pilot.csv
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
LAB_DIR = os.path.join(BASE, "data", "newdb_v2", "labeling_candidates")
NEWDB_DIR = os.path.join(BASE, "data", "newdb_v2")

# ── 1. Load already-labeled IDs (old + all current pilots) ───────────────────
print("=== 1. Building exclude set (already labeled) ===", flush=True)
old_labels = pd.read_csv(os.path.join(BASE, "data", "final", "combined_expert_labels.csv"))
bmk   = pd.read_csv(os.path.join(LAB_DIR, "bmk_pilot_results.csv"))
dt    = pd.read_csv(os.path.join(LAB_DIR, "draa_tafilalet_pilot_results.csv"))

already_labeled = (
    set(old_labels["Locality_ID"])
    | set(bmk[bmk["Expert_Class"].notna()]["Locality_ID"])
    | set(dt[dt["Expert_Class"].notna()]["Locality_ID"])
)
print(f"  Already labeled: {len(already_labeled)}", flush=True)

# ── 2. Load feature-complete sites ───────────────────────────────────────────
feats = pd.read_csv(os.path.join(NEWDB_DIR, "geosites_new_localities_features.csv"))
master = pd.read_csv(os.path.join(NEWDB_DIR, "geosites_localities_master.csv"))
citations = pd.read_csv(os.path.join(LAB_DIR, "all_830_with_citations.csv"))

# Normalise region names (strip whitespace, collapse variants)
REGION_ALIASES = {
    "Beni Mellal-Khenifra":       "Béni Mellal-Khénifra",
    "Beni Mellal-Khénifra":       "Béni Mellal-Khénifra",
    "Béni Mellal-Khénifra ":      "Béni Mellal-Khénifra",
    "Tanger-Al Hoceima":          "Tanger-Al Houceima",
    "Tanger-Tétouan-Al Hoceïma":  "Tanger-Al Houceima",
    " Tanger-Tétouan-Al Hoceïma": "Tanger-Al Houceima",
    "Fès-Meknès":                 "Fés-Meknès",
    "Fés-Meknés":                 "Fés-Meknès",   # accent variant in features CSV
    "Draa-Tafilalet ":            "Draa-Tafilalet",
    "Draa-Tafilalet & Fès-Méknès":"Draa-Tafilalet",
    "Eddakhla-Oued Eddahab":      "Dakhla-Oued Eddahab",
    "Laayoune-Dakhla":            "Dakhla-Oued Eddahab",
    "Laayoune-Sakia El Hamra":    "Dakhla-Oued Eddahab",
    "Marrakech-Saf":              "Marrakech-Safi",
    "Doukkala-Abda":              "Marrakech-Safi",  # closest functional grouping
}

master["Region_norm"] = master["Region"].str.strip().replace(REGION_ALIASES)
feats_with_region = feats.merge(
    master[["Locality_ID", "Region_norm"]], on="Locality_ID", how="left")
feats_with_region["Region_norm"] = feats_with_region["Region_norm"].fillna(
    feats_with_region.get("Region", "Unknown"))

# Merge citations (Reference + Title)
feats_enriched = feats_with_region.merge(
    citations[["Locality_ID", "Reference", "Title"]], on="Locality_ID", how="left")

# Exclude already-labeled
unlabeled = feats_enriched[~feats_enriched["Locality_ID"].isin(already_labeled)].copy()
print(f"  Unlabeled+feature-complete: {len(unlabeled)}", flush=True)
print("  By region:", unlabeled["Region_norm"].value_counts().head(10).to_dict(), flush=True)

# ── 3. Batch output helper ────────────────────────────────────────────────────
OUTPUT_COLS = [
    "Locality_ID", "Geosite_Name", "Region_norm", "Latitude_WGS84", "Longitude_WGS84",
    "Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Highway_m", "LULC_Friction",
    "Dist_to_Settlement_m", "Reference", "Title",
    "Expert_Class", "Confidence", "Expert_Reasoning",
]

def make_batch(regions, n_max, out_filename):
    """Select up to n_max sites from given region list, sorted by road distance."""
    subset = (unlabeled[unlabeled["Region_norm"].isin(regions)]
              .sort_values("Dist_to_Highway_m")
              .head(n_max)
              .copy())
    subset["Expert_Class"] = ""
    subset["Confidence"] = ""
    subset["Expert_Reasoning"] = ""
    # rename Region_norm to Region for readability
    subset = subset.rename(columns={"Region_norm": "Region"})
    out_path = os.path.join(LAB_DIR, out_filename)
    # keep only OUTPUT_COLS that exist
    cols = [c for c in OUTPUT_COLS if c in subset.columns or c in ["Expert_Class","Confidence","Expert_Reasoning"]]
    subset[cols].to_csv(out_path, index=False)
    print(f"  Saved {len(subset)} sites → {out_filename}", flush=True)
    return subset

# ── 4. Generate batches ───────────────────────────────────────────────────────
print("\n=== 2. Generating labeling batches ===", flush=True)

fes = make_batch(
    regions=["Fés-Meknès"],
    n_max=50,
    out_filename="fes_meknes_50_pilot.csv",
)

tanger = make_batch(
    regions=["Tanger-Al Houceima"],
    n_max=50,
    out_filename="tanger_50_pilot.csv",
)

dakhla_rabat = make_batch(
    regions=["Dakhla-Oued Eddahab", "Rabat-Salé-Kénitra"],
    n_max=50,  # likely fewer than 50 in these regions
    out_filename="dakhla_rabat_pilot.csv",
)

marrakech = make_batch(
    regions=["Marrakech-Safi"],
    n_max=50,
    out_filename="marrakech_safi_50_pilot.csv",
)

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===", flush=True)
total_batched = len(fes) + len(tanger) + len(dakhla_rabat) + len(marrakech)
print(f"  Fés-Meknès pilot:    {len(fes)} sites", flush=True)
print(f"  Tanger pilot:        {len(tanger)} sites", flush=True)
print(f"  Dakhla/Rabat pilot:  {len(dakhla_rabat)} sites", flush=True)
print(f"  Marrakech pilot:     {len(marrakech)} sites", flush=True)
print(f"  Total ready for labeling: {total_batched}", flush=True)
print(f"  Remaining unlabeled (not yet batched): "
      f"{len(unlabeled) - total_batched}", flush=True)
print("DONE.", flush=True)
