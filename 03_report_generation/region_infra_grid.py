"""
03_report_generation/region_infra_grid.py

Computes tourism-POI-density and nearest-settlement-type features across a
region's PREDICTION GRID (not just at known geosite points, unlike
02_modeling_and_analysis/16b/16c which built the geosite-level infra_features.csv used in
modeling). Key efficiency point: POIs/settlements are extracted from the OSM
PBF ONCE per region (single pyrosm call, subprocess-isolated + 4GB capped,
same safety pattern as yesterday's geosite-level extraction), then every
grid cell's features are computed via fast vectorized haversine distance --
NOT by re-parsing OSM per grid cell, which is what would make this
prohibitively slow.

Usage: python3 region_infra_grid.py <west> <south> <east> <north> <lon2d_npy> <lat2d_npy> <out_npz> <pbf_path>
  lon2d_npy/lat2d_npy: saved numpy arrays of the prediction grid coordinates
  out_npz: saves n_tourism_poi_10km, dist_nearest_tourism_poi_m,
           dist_nearest_settlement_town_m, settlement_type_code (int, see
           SETTLEMENT_RANK) as a single .npz, same shape as lon2d
"""
import sys, resource
MEM_LIMIT_BYTES = 4 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))

import numpy as np
import pyrosm

SETTLEMENT_TYPES = ["city", "town", "village", "hamlet"]
SETTLEMENT_RANK = {"city": 4, "town": 3, "village": 2, "hamlet": 1}

west, south, east, north, lon2d_path, lat2d_path, out_path, pbf_path = sys.argv[1:9]
west, south, east, north = float(west), float(south), float(east), float(north)

lon2d = np.load(lon2d_path)
lat2d = np.load(lat2d_path)
shape = lon2d.shape

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

n_tourism = np.zeros(shape, dtype=np.float32)
dist_tourism = np.full(shape, np.nan, dtype=np.float32)
dist_settlement = np.full(shape, np.nan, dtype=np.float32)
settle_type_code = np.zeros(shape, dtype=np.int8)

try:
    osm = pyrosm.OSM(pbf_path, bounding_box=[west, south, east, north])

    tourism = osm.get_pois(custom_filter={"tourism": True})
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

    flat_lon, flat_lat = lon2d.ravel(), lat2d.ravel()
    n_flat = len(flat_lon)

    if tourism_pts is not None:
        # chunked to bound memory: distance matrix is n_flat x n_poi
        CHUNK = 2000
        n_flat_arr = np.zeros(n_flat, dtype=np.float32)
        d_flat_arr = np.full(n_flat, np.nan, dtype=np.float32)
        for i0 in range(0, n_flat, CHUNK):
            i1 = min(n_flat, i0 + CHUNK)
            d = haversine(flat_lat[i0:i1, None], flat_lon[i0:i1, None], tourism_pts[None, :, 1], tourism_pts[None, :, 0])
            n_flat_arr[i0:i1] = (d <= 10000).sum(axis=1)
            d_flat_arr[i0:i1] = d.min(axis=1)
        n_tourism = n_flat_arr.reshape(shape)
        dist_tourism = d_flat_arr.reshape(shape)

    if settle_pts is not None:
        CHUNK = 2000
        d_flat_arr = np.full(n_flat, np.nan, dtype=np.float32)
        type_flat_arr = np.zeros(n_flat, dtype=np.int8)
        for i0 in range(0, n_flat, CHUNK):
            i1 = min(n_flat, i0 + CHUNK)
            d = haversine(flat_lat[i0:i1, None], flat_lon[i0:i1, None], settle_pts[None, :, 1], settle_pts[None, :, 0])
            idx = d.argmin(axis=1)
            d_flat_arr[i0:i1] = d[np.arange(i1-i0), idx]
            type_flat_arr[i0:i1] = settle_rank[idx]
        dist_settlement = d_flat_arr.reshape(shape)
        settle_type_code = type_flat_arr.reshape(shape)

    print(f"OK: {len(tourism) if tourism is not None else 0} tourism POIs, "
          f"{len(settlements) if settlements is not None else 0} settlements")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

np.savez(out_path, n_tourism_poi_10km=n_tourism, dist_nearest_tourism_poi_m=dist_tourism,
         dist_nearest_settlement_town_m=dist_settlement, settlement_type_code=settle_type_code)
