"""
Compute Dist_to_Settlement_m: the great-circle (haversine) distance in meters from each
geosite to the nearest of 55 reference Moroccan cities/towns.

Source of the reference settlement list: user-provided "Official Moroccan Localities
Dataset" (data.gov.ma + Wikipedia cross-reference), geocoded via Nominatim (OpenStreetMap)
during the initial catalog-consolidation phase of this project. One bad geocode (Azrou,
originally resolved to 35.13,-3.54 instead of its real location ~33.43,-5.23) was caught
via a regional-plausibility cross-check and hardcoded to the correct coordinate; see
data/archive/pipeline_intermediates/morocco_reference_cities_geocoded.csv for the final
list and geosites_region_mismatches.csv for the correction record.

This recovers a feature that existed in the production dataset (geosites_mcdm_national.csv)
but had no corresponding script in code/ -- flagged as a reproducibility gap by an
independent review (Aug 2026). Re-running this script reproduces that column exactly.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "final")
CITIES_CSV = os.path.join(HERE, "..", "data", "archive", "pipeline_intermediates",
                           "morocco_reference_cities_geocoded.csv")
IN_CSV = os.path.join(DATA_DIR, "geosites_mcdm_national.csv")


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in meters."""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    cities = pd.read_csv(CITIES_CSV)
    print(f"Loaded {len(cities)} reference settlements")
    assert len(cities) == 55, f"Expected 55 reference cities, got {len(cities)} -- source file may have changed"

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} geosites")

    clat, clon = cities["Latitude"].values, cities["Longitude"].values

    def nearest_settlement_dist(row):
        d = haversine_m(row["Latitude_WGS84"], row["Longitude_WGS84"], clat, clon)
        return d.min()

    new_dist = df.apply(nearest_settlement_dist, axis=1)

    if "Dist_to_Settlement_m" in df.columns:
        old = df["Dist_to_Settlement_m"]
        diff = (old - new_dist).abs()
        print(f"Existing column found. Max abs difference from recomputation: {diff.max():.2f} m")
        print(f"Mean abs difference: {diff.mean():.4f} m")
        n_mismatch = (diff > 1.0).sum()
        if n_mismatch > 0:
            print(f"WARNING: {n_mismatch} sites differ by >1m from the recomputed value -- "
                  f"the existing column may have been computed from a different city list.")
        else:
            print("Recomputation matches the existing column exactly (within 1m). Reproducibility confirmed.")
    else:
        print("No existing Dist_to_Settlement_m column found -- adding fresh.")

    df["Dist_to_Settlement_m"] = new_dist
    df.to_csv(IN_CSV, index=False)
    print(f"\nSaved Dist_to_Settlement_m to {IN_CSV}")
    print(df["Dist_to_Settlement_m"].describe())


if __name__ == "__main__":
    main()
