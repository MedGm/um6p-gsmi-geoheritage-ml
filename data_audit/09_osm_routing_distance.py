"""
data_audit/09_osm_routing_distance.py  (2026-08-21, v2 -- chunked)

v1 tried to parse the WHOLE Morocco OSM extract as one graph
(network_type="all") -- OOM-killed at ~10.8GB anon-rss (this machine's
container caps well under the host's 15GB). Rewritten to process in
geographic grid cells: for each occupied 1.5-degree cell (+0.3deg buffer),
read only that bbox from the PBF (pyrosm supports this natively, no repeat
downloads), build a small local graph, run multi-source Dijkstra to the
nearest paved-highway node within that cell only, then free the graph
before moving to the next cell. Keeps peak memory bounded regardless of
country size.

Builds a true road-NETWORK routing distance feature, as opposed to the
existing Dist_to_Highway_m (straight-line point-to-nearest-line distance,
paved classes only: motorway/trunk/primary/secondary/tertiary, see code/02).
Two prior "OSM-aware" attempts (G0b straight-line-to-trail, LCP terrain-
friction least-cost-path) gave mixed/marginal results -- see
code/27_lcp_feature_test.py docstring. Neither used real road-network
connectivity; this does.

Output: data/final/dist_to_highway_routing_m.csv
"""
import gc, os, time, warnings
import numpy as np, pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
import pyrosm

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
PBF_PATH = os.path.join(BASE, "data/osm/morocco-latest.osm.pbf")
PAVED_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary",
                  "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}

CELL_DEG = 1.5   # grid cell size
BUFFER_DEG = 0.3  # extra margin read around each cell so routes aren't truncated at the edge

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
sites = catalog[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84", "Dist_to_Highway_m"]].dropna(
    subset=["Latitude_WGS84", "Longitude_WGS84"]).reset_index(drop=True)
log(f"{len(sites)} geosites with valid coordinates")

sites["cell_lat"] = (sites["Latitude_WGS84"] // CELL_DEG).astype(int)
sites["cell_lon"] = (sites["Longitude_WGS84"] // CELL_DEG).astype(int)
cells = sites.groupby(["cell_lat", "cell_lon"])
log(f"{cells.ngroups} occupied grid cells")

all_results = []
for i, ((clat, clon), group) in enumerate(cells):
    south = clat * CELL_DEG - BUFFER_DEG
    north = (clat + 1) * CELL_DEG + BUFFER_DEG
    west = clon * CELL_DEG - BUFFER_DEG
    east = (clon + 1) * CELL_DEG + BUFFER_DEG
    log(f"[{i+1}/{cells.ngroups}] cell ({clat},{clon}) n_sites={len(group)} bbox=[{west:.2f},{south:.2f},{east:.2f},{north:.2f}]")

    try:
        osm = pyrosm.OSM(PBF_PATH, bounding_box=[west, south, east, north])
        result = osm.get_network(network_type="all", nodes=True)
        if result is None or result[0] is None or len(result[0]) == 0:
            log("  no road data in this cell -- skipping (routing distance = NaN)")
            for lid in group["Locality_ID"]:
                all_results.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": np.nan})
            del osm
            gc.collect()
            continue
        nodes, edges = result
        G = osm.to_graph(nodes, edges, graph_type="networkx")

        paved_nodes = set()
        for u, v, data in G.edges(data=True):
            hwy = data.get("highway")
            if isinstance(hwy, list):
                hwy = hwy[0] if hwy else None
            if hwy in PAVED_CLASSES:
                paved_nodes.add(u); paved_nodes.add(v)

        if not paved_nodes:
            log("  no paved-class roads in this cell -- skipping (routing distance = NaN)")
            for lid in group["Locality_ID"]:
                all_results.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": np.nan})
            del osm, G
            gc.collect()
            continue

        Gu = G.to_undirected()
        dist_to_paved = nx.multi_source_dijkstra_path_length(Gu, sources=paved_nodes, weight="length")

        node_ids = np.array(list(G.nodes()))
        node_lat = np.array([G.nodes[n]["y"] for n in node_ids])
        node_lon = np.array([G.nodes[n]["x"] for n in node_ids])
        tree = cKDTree(np.column_stack([node_lon, node_lat]))

        _, idx = tree.query(group[["Longitude_WGS84", "Latitude_WGS84"]].values, k=1)
        nearest_nodes = node_ids[idx]
        for lid, nn in zip(group["Locality_ID"], nearest_nodes):
            all_results.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": dist_to_paved.get(nn, np.nan)})

        log(f"  cell done: {G.number_of_nodes()} nodes, {len(paved_nodes)} paved nodes, "
            f"{sum(1 for lid,nn in zip(group['Locality_ID'],nearest_nodes) if dist_to_paved.get(nn) is not None)} sites resolved")

        del osm, G, Gu, dist_to_paved, tree
        gc.collect()

    except Exception as e:
        log(f"  CELL FAILED: {type(e).__name__}: {e} -- marking sites as NaN")
        for lid in group["Locality_ID"]:
            all_results.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": np.nan})
        gc.collect()

out = pd.DataFrame(all_results)
out_path = os.path.join(BASE, "data/final/dist_to_highway_routing_m.csv")
out.to_csv(out_path, index=False)
log(f"Saved {out_path} ({out['Dist_to_Highway_Routing_m'].notna().sum()}/{len(out)} resolved)")

merged = sites.merge(out, on="Locality_ID", how="left")
valid = merged.dropna(subset=["Dist_to_Highway_Routing_m"])
ratio = valid["Dist_to_Highway_Routing_m"] / valid["Dist_to_Highway_m"].clip(lower=1)
log(f"routing/straight-line ratio: median={ratio.median():.2f} mean={ratio.mean():.2f} (>=1 expected)")
log(f"sites where routing < straight-line (tolerance for snap error): {(ratio < 0.98).sum()} / {len(valid)}")
