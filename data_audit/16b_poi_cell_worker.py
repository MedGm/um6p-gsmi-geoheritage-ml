"""
data_audit/16b_poi_cell_worker.py

Processes ONE grid cell: extracts tourism/amenity POIs and settlement
(place) nodes from that bbox only, computes per-site infrastructure
features, writes to CSV, exits. Same isolation pattern as
09b_osm_routing_cell_worker.py (own subprocess, hard memory cap) --
a raw whole-country osm.get_pois() call (no bbox, no cap) OOM-killed the
host machine today exactly like the unbounded routing-graph call did
earlier. Never call pyrosm on the full PBF without a bounding_box again.

Usage: python3 16b_poi_cell_worker.py <clat> <clon> <cell_deg> <buffer_deg> <sites_csv> <out_csv> <pbf_path>
  sites_csv: Locality_ID,Latitude_WGS84,Longitude_WGS84 for just this cell's sites
  out_csv: Locality_ID, n_tourism_poi_10km, dist_nearest_tourism_poi_m,
           dist_nearest_settlement_town_m, nearest_settlement_type
"""
import sys, resource
MEM_LIMIT_BYTES = 4 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))

import numpy as np, pandas as pd
import pyrosm
from scipy.spatial import cKDTree

TOURISM_TAGS = {"tourism": True}
SETTLEMENT_TYPES = ["city", "town", "village", "hamlet"]
SETTLEMENT_RANK = {"city": 4, "town": 3, "village": 2, "hamlet": 1}

clat, clon, cell_deg, buffer_deg, sites_csv, out_csv, pbf_path = sys.argv[1:8]
clat, clon, cell_deg, buffer_deg = int(clat), int(clon), float(cell_deg), float(buffer_deg)

group = pd.read_csv(sites_csv)
south = clat * cell_deg - buffer_deg
north = (clat + 1) * cell_deg + buffer_deg
west = clon * cell_deg - buffer_deg
east = (clon + 1) * cell_deg + buffer_deg

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

rows = []
try:
    osm = pyrosm.OSM(pbf_path, bounding_box=[west, south, east, north])

    tourism = osm.get_pois(custom_filter=TOURISM_TAGS)
    tourism_pts = None
    if tourism is not None and len(tourism) > 0:
        tourism = tourism[tourism.geometry.notna()]
        cent = tourism.geometry.centroid
        tourism_pts = np.column_stack([cent.x.values, cent.y.values])

    settlements = osm.get_pois(custom_filter={"place": SETTLEMENT_TYPES})
    settle_pts, settle_rank = None, None
    if settlements is not None and len(settlements) > 0:
        settlements = settlements[settlements.geometry.notna() & settlements["place"].isin(SETTLEMENT_TYPES)]
        if len(settlements) > 0:
            cent = settlements.geometry.centroid
            settle_pts = np.column_stack([cent.x.values, cent.y.values])
            settle_rank = settlements["place"].map(SETTLEMENT_RANK).values

    for _, site in group.iterrows():
        lat0, lon0 = site["Latitude_WGS84"], site["Longitude_WGS84"]
        n_tourism_10km, dist_tourism = 0, np.nan
        if tourism_pts is not None:
            d = haversine(lat0, lon0, tourism_pts[:, 1], tourism_pts[:, 0])
            n_tourism_10km = int((d <= 10000).sum())
            dist_tourism = float(d.min())
        dist_settlement, settlement_type = np.nan, None
        if settle_pts is not None:
            d = haversine(lat0, lon0, settle_pts[:, 1], settle_pts[:, 0])
            i = np.argmin(d)
            dist_settlement = float(d[i])
            settlement_type = list(SETTLEMENT_RANK.keys())[list(SETTLEMENT_RANK.values()).index(settle_rank[i])] if settle_rank[i] in SETTLEMENT_RANK.values() else None
        rows.append({
            "Locality_ID": site["Locality_ID"],
            "n_tourism_poi_10km": n_tourism_10km,
            "dist_nearest_tourism_poi_m": dist_tourism,
            "dist_nearest_settlement_town_m": dist_settlement,
            "nearest_settlement_type": settlement_type,
        })
    print(f"cell done: {len(tourism) if tourism is not None else 0} tourism POIs, "
          f"{len(settlements) if settlements is not None else 0} settlements")
except Exception as e:
    print(f"CELL FAILED: {type(e).__name__}: {e}")
    for _, site in group.iterrows():
        rows.append({"Locality_ID": site["Locality_ID"], "n_tourism_poi_10km": None,
                      "dist_nearest_tourism_poi_m": None, "dist_nearest_settlement_town_m": None,
                      "nearest_settlement_type": None})

pd.DataFrame(rows).to_csv(out_csv, index=False)
