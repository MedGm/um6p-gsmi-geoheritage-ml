"""
data_audit/09c_osm_routing_driver.py  (v2 -- smaller cells, hard memory cap)

Resumable driver: loops over grid cells, calls 09b_osm_routing_cell_worker.py
as a fresh subprocess per cell. Each worker now also self-enforces a hard
2GB RLIMIT_AS (see 09b's docstring) -- process isolation alone still let one
dense cell hit ~10GB and crash the whole host machine, not just get
container-OOM-killed. Cells shrunk from 1.5deg/0.3deg buffer to
0.4deg/0.1deg buffer (~14x smaller area) as a second, independent mitigation
on top of the hard cap.

Reuses any site already resolved by a PRE-v2 checkpoint (the 1.5deg-grid
run that got interrupted) by Locality_ID, regardless of the old grid scheme
-- that ~20min of work isn't thrown away, only the unresolved sites get
regridded and reprocessed under the new, safer cell size.

Output: data/final/dist_to_highway_routing_m.csv, plus per-cell checkpoint
files in data_audit/logs/osm_cells/ (safe to delete after a successful run).
"""
import glob, os, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
PBF_PATH = os.path.join(BASE, "data/osm/morocco-latest.osm.pbf")
CELL_DIR = os.path.join(BASE, "data_audit/logs/osm_cells")
os.makedirs(CELL_DIR, exist_ok=True)
CELL_DEG, BUFFER_DEG = 0.4, 0.1

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
sites = catalog[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84", "Dist_to_Highway_m"]].dropna(
    subset=["Latitude_WGS84", "Longitude_WGS84"]).reset_index(drop=True)

# --- Reuse sites already resolved by the pre-v2 (1.5deg grid) checkpoints ---
old_checkpoints = [f for f in glob.glob(os.path.join(CELL_DIR, "cell_*.csv"))]
already_resolved = pd.DataFrame(columns=["Locality_ID", "Dist_to_Highway_Routing_m"])
if old_checkpoints:
    already_resolved = pd.concat([pd.read_csv(f) for f in old_checkpoints], ignore_index=True)
    already_resolved = already_resolved.dropna(subset=["Dist_to_Highway_Routing_m"]).drop_duplicates("Locality_ID")
    log(f"reusing {len(already_resolved)} sites already resolved by the pre-v2 run")

remaining = sites[~sites["Locality_ID"].isin(already_resolved["Locality_ID"])].reset_index(drop=True)
log(f"{len(sites)} total geosites, {len(remaining)} still need routing distance (0.4deg grid, 4GB/worker hard cap)")

remaining["cell_lat"] = (remaining["Latitude_WGS84"] // CELL_DEG).astype(int)
remaining["cell_lon"] = (remaining["Longitude_WGS84"] // CELL_DEG).astype(int)
cells = list(remaining.groupby(["cell_lat", "cell_lon"]))
log(f"{len(cells)} occupied cells at the new grid size")

for i, ((clat, clon), group) in enumerate(cells):
    cell_out = os.path.join(CELL_DIR, f"v2_cell_{clat}_{clon}.csv")
    if os.path.exists(cell_out):
        log(f"[{i+1}/{len(cells)}] cell ({clat},{clon}) already done -- skipping")
        continue
    sites_csv = os.path.join(CELL_DIR, f"_input_v2_{clat}_{clon}.csv")
    group[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"]].to_csv(sites_csv, index=False)
    log(f"[{i+1}/{len(cells)}] cell ({clat},{clon}) n_sites={len(group)} -- launching worker subprocess")
    r = subprocess.run([sys.executable, os.path.join(HERE, "09b_osm_routing_cell_worker.py"),
                         str(clat), str(clon), str(CELL_DEG), str(BUFFER_DEG), sites_csv, cell_out, PBF_PATH],
                        capture_output=True, text=True, timeout=600)
    for line in (r.stdout + r.stderr).splitlines():
        log(f"  worker: {line}")
    if r.returncode != 0 or not os.path.exists(cell_out):
        log(f"  WORKER CRASHED/CAPPED (returncode={r.returncode}) -- writing NaN fallback for this cell's sites")
        pd.DataFrame({"Locality_ID": group["Locality_ID"], "Dist_to_Highway_Routing_m": None}).to_csv(cell_out, index=False)
    os.remove(sites_csv)

log("All cells done, merging ...")
all_new = glob.glob(os.path.join(CELL_DIR, "v2_cell_*.csv"))
new_results = pd.concat([pd.read_csv(f) for f in all_new], ignore_index=True) if all_new else pd.DataFrame(
    columns=["Locality_ID", "Dist_to_Highway_Routing_m"])
out = pd.concat([already_resolved, new_results], ignore_index=True).drop_duplicates("Locality_ID")
out_path = os.path.join(BASE, "data/final/dist_to_highway_routing_m.csv")
out.to_csv(out_path, index=False)
log(f"Saved {out_path} ({out['Dist_to_Highway_Routing_m'].notna().sum()}/{len(sites)} resolved)")

merged = sites.merge(out, on="Locality_ID", how="left")
valid = merged.dropna(subset=["Dist_to_Highway_Routing_m"])
ratio = valid["Dist_to_Highway_Routing_m"] / valid["Dist_to_Highway_m"].clip(lower=1)
log(f"routing/straight-line ratio: median={ratio.median():.2f} mean={ratio.mean():.2f} (>=1 expected)")
