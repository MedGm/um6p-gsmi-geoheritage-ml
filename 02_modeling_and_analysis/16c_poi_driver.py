"""
data_audit/16c_poi_driver.py

Resumable chunked driver for the tourism-POI / settlement-type infrastructure
features, same pattern as 09c_osm_routing_driver.py: subprocess-per-cell
(16b_poi_cell_worker.py), 0.4deg grid + 0.1deg buffer (already proven safe
for this machine at that size), checkpoint files so a crash only costs
cells since the last checkpoint.

Output: data/final/infra_features.csv
  Locality_ID, n_tourism_poi_10km, dist_nearest_tourism_poi_m,
  dist_nearest_settlement_town_m, nearest_settlement_type
"""
import glob, os, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
PBF_PATH = os.path.join(BASE, "data/osm/morocco-latest.osm.pbf")
CELL_DIR = os.path.join(BASE, "data_audit/logs/poi_cells")
os.makedirs(CELL_DIR, exist_ok=True)
CELL_DEG, BUFFER_DEG = 0.4, 0.1

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
sites = catalog[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"]].dropna(
    subset=["Latitude_WGS84", "Longitude_WGS84"]).reset_index(drop=True)
sites["cell_lat"] = (sites["Latitude_WGS84"] // CELL_DEG).astype(int)
sites["cell_lon"] = (sites["Longitude_WGS84"] // CELL_DEG).astype(int)
cells = list(sites.groupby(["cell_lat", "cell_lon"]))
log(f"{len(sites)} geosites, {len(cells)} occupied cells (0.4deg grid)")

for i, ((clat, clon), group) in enumerate(cells):
    cell_out = os.path.join(CELL_DIR, f"cell_{clat}_{clon}.csv")
    if os.path.exists(cell_out):
        log(f"[{i+1}/{len(cells)}] cell ({clat},{clon}) already done -- skipping")
        continue
    sites_csv = os.path.join(CELL_DIR, f"_input_{clat}_{clon}.csv")
    group[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"]].to_csv(sites_csv, index=False)
    log(f"[{i+1}/{len(cells)}] cell ({clat},{clon}) n_sites={len(group)} -- launching worker")
    r = subprocess.run([sys.executable, os.path.join(HERE, "16b_poi_cell_worker.py"),
                         str(clat), str(clon), str(CELL_DEG), str(BUFFER_DEG), sites_csv, cell_out, PBF_PATH],
                        capture_output=True, text=True, timeout=600)
    for line in (r.stdout + r.stderr).splitlines():
        log(f"  worker: {line}")
    if r.returncode != 0 or not os.path.exists(cell_out):
        log(f"  WORKER FAILED (returncode={r.returncode}) -- writing NaN fallback")
        pd.DataFrame({"Locality_ID": group["Locality_ID"], "n_tourism_poi_10km": None,
                      "dist_nearest_tourism_poi_m": None, "dist_nearest_settlement_town_m": None,
                      "nearest_settlement_type": None}).to_csv(cell_out, index=False)
    os.remove(sites_csv)

log("All cells done, merging ...")
all_files = glob.glob(os.path.join(CELL_DIR, "cell_*.csv"))
out = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
out_path = os.path.join(BASE, "data/final/infra_features.csv")
out.to_csv(out_path, index=False)
log(f"Saved {out_path} ({out['n_tourism_poi_10km'].notna().sum()}/{len(sites)} resolved)")
log(f"\nTourism POI count distribution:\n{out['n_tourism_poi_10km'].describe()}")
log(f"\nSettlement type distribution:\n{out['nearest_settlement_type'].value_counts()}")
