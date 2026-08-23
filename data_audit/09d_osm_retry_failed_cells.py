"""
data_audit/09d_osm_retry_failed_cells.py

Targeted retry for the 27 cells that hit the 4GB worker memory cap in the
0.4deg run (09c) -- concentrated in Tanger-Tétouan-Al Hoceima (107/128
labeled sites missing routing distance) and Fés-Meknés (90/337 missing),
the two densest/most important regions. Re-grids ONLY those failed cells'
sites at a finer 0.15deg sub-grid (vs the original 0.4deg) so each piece is
small enough to resolve under the same 4GB cap, using the same isolated
subprocess worker (09b) unchanged.

Output: merges into data/final/dist_to_highway_routing_m.csv, filling in
previously-NaN sites where the finer grid now resolves them.
"""
import glob, os, re, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
PBF_PATH = os.path.join(BASE, "data/osm/morocco-latest.osm.pbf")
CELL_DIR = os.path.join(BASE, "data_audit/logs/osm_cells")
LOG_PATH = os.path.join(BASE, "data_audit/logs/osm_routing.log")

ORIG_CELL_DEG = 0.4
SUB_CELL_DEG = 0.15
SUB_BUFFER_DEG = 0.05

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# --- Find the failed (clat, clon) cells from the 09c log ---
lines = open(LOG_PATH).read().splitlines()
failed_cells = []
for i, line in enumerate(lines):
    if "CELL FAILED: MemoryError" in line:
        for j in range(i - 1, -1, -1):
            m = re.search(r"cell \((-?\d+),(-?\d+)\)", lines[j])
            if m:
                failed_cells.append((int(m.group(1)), int(m.group(2))))
                break
log(f"{len(failed_cells)} failed cells to retry: {failed_cells}")

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
sites = catalog[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84", "Dist_to_Highway_m"]].dropna(
    subset=["Latitude_WGS84", "Longitude_WGS84"]).reset_index(drop=True)
sites["orig_clat"] = (sites["Latitude_WGS84"] // ORIG_CELL_DEG).astype(int)
sites["orig_clon"] = (sites["Longitude_WGS84"] // ORIG_CELL_DEG).astype(int)

failed_set = set(failed_cells)
retry_sites = sites[sites.apply(lambda r: (r.orig_clat, r.orig_clon) in failed_set, axis=1)].copy()
log(f"{len(retry_sites)} sites belong to failed cells, regridding at {SUB_CELL_DEG}deg")

retry_sites["sub_clat"] = (retry_sites["Latitude_WGS84"] // SUB_CELL_DEG).astype(int)
retry_sites["sub_clon"] = (retry_sites["Longitude_WGS84"] // SUB_CELL_DEG).astype(int)
sub_cells = list(retry_sites.groupby(["sub_clat", "sub_clon"]))
log(f"{len(sub_cells)} finer sub-cells to process")

for i, ((clat, clon), group) in enumerate(sub_cells):
    cell_out = os.path.join(CELL_DIR, f"retry_cell_{clat}_{clon}.csv")
    if os.path.exists(cell_out):
        log(f"[{i+1}/{len(sub_cells)}] sub-cell ({clat},{clon}) already done -- skipping")
        continue
    sites_csv = os.path.join(CELL_DIR, f"_input_retry_{clat}_{clon}.csv")
    group[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"]].to_csv(sites_csv, index=False)
    log(f"[{i+1}/{len(sub_cells)}] sub-cell ({clat},{clon}) n_sites={len(group)} -- launching worker")
    r = subprocess.run([sys.executable, os.path.join(HERE, "09b_osm_routing_cell_worker.py"),
                         str(clat), str(clon), str(SUB_CELL_DEG), str(SUB_BUFFER_DEG), sites_csv, cell_out, PBF_PATH],
                        capture_output=True, text=True, timeout=600)
    for line in (r.stdout + r.stderr).splitlines():
        log(f"  worker: {line}")
    if r.returncode != 0 or not os.path.exists(cell_out):
        log(f"  STILL FAILED (returncode={r.returncode}) -- writing NaN fallback")
        pd.DataFrame({"Locality_ID": group["Locality_ID"], "Dist_to_Highway_Routing_m": None}).to_csv(cell_out, index=False)
    os.remove(sites_csv)

log("Merging retry results into final output ...")
retry_files = glob.glob(os.path.join(CELL_DIR, "retry_cell_*.csv"))
retry_results = pd.concat([pd.read_csv(f) for f in retry_files], ignore_index=True) if retry_files else pd.DataFrame(
    columns=["Locality_ID", "Dist_to_Highway_Routing_m"])
retry_results = retry_results.dropna(subset=["Dist_to_Highway_Routing_m"])
log(f"retry resolved {len(retry_results)} / {len(retry_sites)} previously-failed sites")

out_path = os.path.join(BASE, "data/final/dist_to_highway_routing_m.csv")
existing = pd.read_csv(out_path)
existing = existing.set_index("Locality_ID")
retry_results = retry_results.set_index("Locality_ID")
existing.update(retry_results)  # fills in previously-NaN rows where retry succeeded
existing = existing.reset_index()
existing.to_csv(out_path, index=False)
log(f"Saved {out_path} ({existing['Dist_to_Highway_Routing_m'].notna().sum()}/{len(sites)} resolved overall)")

merged = sites.merge(existing, on="Locality_ID", how="left")
valid = merged.dropna(subset=["Dist_to_Highway_Routing_m"])
ratio = valid["Dist_to_Highway_Routing_m"] / valid["Dist_to_Highway_m"].clip(lower=1)
log(f"routing/straight-line ratio: median={ratio.median():.2f} mean={ratio.mean():.2f}")
