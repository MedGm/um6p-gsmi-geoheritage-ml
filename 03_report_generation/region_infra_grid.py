"""
03_report_generation/region_infra_grid.py

Computes tourism-POI-density and nearest-settlement-type features across a
region's PREDICTION GRID (not just at known geosite points, unlike
02_modeling_and_analysis/16b/16c which built the geosite-level infra_features.csv used in
modeling).

2026-09-01: rewritten from pyrosm's bbox-filtered get_pois() (GeoDataFrame +
geometry/centroid machinery) to pyosmium's streaming node handler. The old
approach OOM'd (and once outright segfaulted under a memory cap) on
quarter-country-scale bboxes needed for a national-resolution grid -- dense
tiles like the Marrakech/Atlas corridor never completed even at an 8GB cap.
pyosmium streams the whole 232MB national PBF once, keeping only matching
node coordinates (a few thousand points, not the whole road/building
network): confirmed processing the ENTIRE country in ~98s. Memory is still
several GB (pyosmium's own node-location index covers every node in the
file, not just matches -- a 2GB cap triggered std::bad_alloc; 6GB is safe
with margin), but is fixed and predictable, unlike pyrosm's per-bbox-size
scaling which OOM'd (and once segfaulted under a memory cap) on
quarter-country-scale bboxes -- dense tiles like the Marrakech/Atlas corridor
never completed even at an 8GB cap. Always extracts nationally regardless
of the requested bbox (cheap enough that per-region bbox-limiting isn't worth
the complexity) -- the bbox args are kept for interface compatibility with
existing callers but no longer used to scope the OSM read itself.

Node-only simplification: tourism/place tags in OSM are overwhelmingly
point-like in practice (a hotel, a village) -- way/relation-tagged instances
(e.g. a settlement mapped as an administrative boundary polygon) are not
captured. This trades a small amount of recall for the segfault-proof
streaming approach; same trade-off already implicitly made by the geometry
centroid step in the old code for anything not cleanly point-like.

Usage: python3 region_infra_grid.py <west> <south> <east> <north> <lon2d_npy> <lat2d_npy> <out_npz> <pbf_path>
  lon2d_npy/lat2d_npy: saved numpy arrays of the prediction grid coordinates
  out_npz: saves n_tourism_poi_10km, dist_nearest_tourism_poi_m,
           dist_nearest_settlement_town_m, settlement_type_code (int, see
           SETTLEMENT_RANK) as a single .npz, same shape as lon2d
"""
import sys, resource
# pyosmium's location index (resolving every node's coordinates, not just
# matches) needs several GB for the whole country -- confirmed a 2GB cap
# triggers std::bad_alloc even though matched-point storage itself is tiny.
# An uncapped run completed fine within ~11GB of headroom on this machine.
MEM_LIMIT_BYTES = 6 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))

import numpy as np
import osmium

SETTLEMENT_TYPES = {"city", "town", "village", "hamlet"}
SETTLEMENT_RANK = {"city": 4, "town": 3, "village": 2, "hamlet": 1}

west, south, east, north, lon2d_path, lat2d_path, out_path, pbf_path = sys.argv[1:9]

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

class POIHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.tourism_pts = []
        self.settle_pts = []
        self.settle_ranks = []

    def node(self, n):
        tags = n.tags
        if "tourism" in tags:
            self.tourism_pts.append((n.location.lon, n.location.lat))
        place = tags.get("place")
        if place in SETTLEMENT_TYPES:
            self.settle_pts.append((n.location.lon, n.location.lat))
            self.settle_ranks.append(SETTLEMENT_RANK[place])

try:
    handler = POIHandler()
    handler.apply_file(pbf_path, locations=True)
    tourism_pts = np.array(handler.tourism_pts) if handler.tourism_pts else None
    settle_pts = np.array(handler.settle_pts) if handler.settle_pts else None
    settle_rank = np.array(handler.settle_ranks) if handler.settle_ranks else None

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

    print(f"OK: {len(handler.tourism_pts)} tourism POIs, {len(handler.settle_pts)} settlements")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

np.savez(out_path, n_tourism_poi_10km=n_tourism, dist_nearest_tourism_poi_m=dist_tourism,
         dist_nearest_settlement_town_m=dist_settlement, settlement_type_code=settle_type_code)
