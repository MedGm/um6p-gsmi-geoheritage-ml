"""
Task 5b: Draft accessibility labels from OSRM real-world driving travel time.

Generates a DRAFT accessibility class for each geosite locality in
geosites_features.csv, using OSRM (public router.project-osrm.org) driving
travel time to the nearest Moroccan reference city as the raw signal.
Output is explicitly a draft for human review (see geosites_labels_draft.csv
columns User_Override_Class / User_Notes, intentionally left empty here).

Reuses the API-call pattern (caching, retry/backoff, nearest-N-cities
pre-filter, politeness delay) from the earlier project attempt at
livrable/phase1_national_accessibility/code/02_generate_accessibility_labels.py.

Task 5b fix (this version) vs. the original Task 5 script:
1. Reference-city set expanded from 12 major metros to a verified, geocoded
   55-city set (morocco_reference_cities_geocoded.csv) spanning all 12 regions
   and including mid-size regional towns, not just the largest metros. The old
   12-city set was too sparse and mislabeled at least one known-easy site
   (Punta Cires, field-verified as an 8-minute local drive, showed 62 min to
   the nearest of the 12 cities).
2. Class thresholds switched from percentile-anchored (data-informed, e.g.
   "Easy" meaning "under 95 min") to FIXED, absolute, real-world-meaningful
   driving-time bands: Easy <= 45 min, Moderate 45-90 min, Difficult 90-180
   min, Very Difficult > 180 min. These are not adjusted to force a nicer
   distribution -- whatever distribution results is reported honestly.
"""
import os, json, time, hashlib, math
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(HERE, "..", "data", "training", "geosites_features.csv")
OUT_CSV = os.path.join(HERE, "..", "data", "training", "geosites_labels_draft.csv")
REFERENCE_CITIES_CSV = os.path.join(
    HERE, "..", "data", "pipeline_intermediates", "morocco_reference_cities_geocoded.csv"
)
CACHE_DIR = os.path.join(HERE, "..", ".osrm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
REQUEST_DELAY_S = 1.0   # be polite to the public demo server
N_QUERY_CITIES = 7      # query only the N geographically-nearest reference cities per site
                         # (raised from 4 -> 7 now that the reference set is 55 cities, not 12,
                         # so the haversine pre-filter still has enough candidates to find the
                         # true nearest-by-road city, not just nearest-by-straight-line)

# Verified, geocoded reference cities (55 cities spanning all 12 regions; see
# morocco_reference_cities_geocoded.csv for provenance / verification notes).
_ref_df = pd.read_csv(REFERENCE_CITIES_CSV)
REFERENCE_CITIES = {
    row["City"]: (row["Latitude"], row["Longitude"]) for _, row in _ref_df.iterrows()
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cache_key(lat, lon, dlat, dlon):
    return hashlib.md5(f"{lat:.5f}_{lon:.5f}_{dlat:.5f}_{dlon:.5f}".encode()).hexdigest()


def osrm_travel_time_minutes(lat, lon, dlat, dlon, retries=3):
    key = cache_key(lat, lon, dlat, dlon)
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)["minutes"]
        return cached, True  # (minutes, was_cached) -- None is a valid cached "no route" result

    url = f"{OSRM_URL}/{lon},{lat};{dlon},{dlat}?overview=false"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            minutes = None
            if data.get("code") == "Ok":
                minutes = data["routes"][0]["duration"] / 60.0
            with open(cache_path, "w") as f:
                json.dump({"minutes": minutes}, f)
            return minutes, False
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None, False


def main():
    df = pd.read_csv(FEATURES_CSV)
    n_sites = len(df)
    print(f"Computing OSRM travel time for {n_sites} geosites "
          f"(querying nearest {N_QUERY_CITIES} of {len(REFERENCE_CITIES)} reference cities each)...")

    min_minutes = np.full(n_sites, np.nan)
    n_requests = 0
    for i, row in df.iterrows():
        lat, lon = row["Latitude_WGS84"], row["Longitude_WGS84"]
        ranked = sorted(
            REFERENCE_CITIES.items(),
            key=lambda kv: haversine_km(lat, lon, kv[1][0], kv[1][1]),
        )[:N_QUERY_CITIES]

        best = np.inf
        for city, (dlat, dlon) in ranked:
            m, was_cached = osrm_travel_time_minutes(lat, lon, dlat, dlon)
            if not was_cached:
                n_requests += 1
                time.sleep(REQUEST_DELAY_S)
            if m is not None and m < best:
                best = m
        min_minutes[i] = best if best != np.inf else np.nan
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_sites} sites processed ({n_requests} live OSRM requests so far)")

    df["OSRM_Travel_Time_Min"] = min_minutes
    n_missing = df["OSRM_Travel_Time_Min"].isna().sum()
    print(f"\nLive OSRM requests made this run: {n_requests}")
    print(f"Missing OSRM routes: {n_missing} / {n_sites}")
    if n_missing > 0:
        if n_missing >= n_sites * 0.15:
            raise RuntimeError(
                f"Too many failed OSRM routes ({n_missing}/{n_sites}) -- check connectivity / rate limiting. "
                "Refusing to fabricate travel times for these sites."
            )
        med = df["OSRM_Travel_Time_Min"].median()
        print(f"Imputing {n_missing} missing travel times with the dataset median ({med:.1f} min).")
        df["OSRM_Travel_Time_Min"] = df["OSRM_Travel_Time_Min"].fillna(med)

    # --- Print the real distribution before picking any thresholds -----------------------
    pct = df["OSRM_Travel_Time_Min"].quantile([0, 0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 1.0])
    print("\n--- Raw OSRM_Travel_Time_Min distribution (minutes) ---")
    for q, v in pct.items():
        print(f"  {int(q*100):>3}th pct: {v:7.1f} min")

    # --- Threshold selection: FIXED, absolute, real-world-meaningful bands ----------------
    # Not percentile/data-anchored. These are literal driving-time bands so "Easy" always
    # means "genuinely short drive" regardless of how the underlying site population happens
    # to be distributed. Reported honestly even if the resulting distribution is skewed.
    t_easy, t_moderate, t_difficult = 45.0, 90.0, 180.0
    print(f"\nFixed thresholds (absolute, not data-informed): "
          f"Easy <= {t_easy:.0f} min, Moderate <= {t_moderate:.0f} min, "
          f"Difficult <= {t_difficult:.0f} min, Very Difficult > {t_difficult:.0f} min")

    bins = [-0.1, t_easy, t_moderate, t_difficult, np.inf]
    labels = ["Easy", "Moderate", "Difficult", "Very Difficult"]
    df["Draft_Accessibility_Class"] = pd.cut(df["OSRM_Travel_Time_Min"], bins=bins, labels=labels)
    assert df["Draft_Accessibility_Class"].isna().sum() == 0, "Some travel times fell outside the bin edges"

    print("\n--- Draft class distribution (fixed absolute thresholds, reported as-is) ---")
    dist = df["Draft_Accessibility_Class"].value_counts().reindex(labels)
    print(dist)

    # --- Verify known-easy reference sites --------------------------------------------------
    check_names = ["Cap Malabata", "Fahs-Anjra (Melloussa)", "Punta Cires", "flute casts of Punta Cires"]
    print("\n--- Known-easy reference site check ---")
    found_any_cires = False
    for name in ["Cap Malabata", "Fahs-Anjra (Melloussa)"]:
        rows = df[df["Geosite_Name"] == name]
        if rows.empty:
            print(f"  {name}: NOT FOUND in geosites_features.csv")
        else:
            for _, r in rows.iterrows():
                print(f"  {name} ({r['Locality_ID']}): {r['OSRM_Travel_Time_Min']:.1f} min -> {r['Draft_Accessibility_Class']}")
    cires_rows = df[df["Geosite_Name"].str.contains("Cires", case=False, na=False)]
    if cires_rows.empty:
        print("  Punta Cires: NOT FOUND in geosites_features.csv (no coordinates / no matching row)")
    else:
        found_any_cires = True
        for _, r in cires_rows.iterrows():
            print(f"  {r['Geosite_Name']} ({r['Locality_ID']}): {r['OSRM_Travel_Time_Min']:.1f} min -> {r['Draft_Accessibility_Class']}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_cols = ["Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Region",
                "OSRM_Travel_Time_Min", "Draft_Accessibility_Class"]
    out = df[out_cols].copy()
    out["User_Override_Class"] = ""
    out["User_Notes"] = ""
    out = out.sort_values(["Region", "Geosite_Name"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved draft labels to {OUT_CSV}")


if __name__ == "__main__":
    main()
