"""
data_audit/09b_osm_routing_cell_worker.py

Processes ONE grid cell and writes its result to a per-cell CSV, then exits.
Meant to be invoked as a subprocess by 09c_osm_routing_driver.py -- running
each cell in its own process guarantees the OS reclaims ALL memory (pyrosm/
GEOS/networkx C-level allocations included) when the cell finishes, which
gc.collect() alone doesn't reliably do within one long-lived process (v2 of
09_osm_routing_distance.py showed RSS climbing steadily cell over cell:
~300MB -> 6.4GB by cell 12/31, real risk of repeating the earlier OOM kill
before reaching the denser northern cells like Fés-Meknés/491 sites).

Hard memory ceiling added after the isolated-subprocess version STILL brought
down the whole host machine (not just a container OOM-kill this time) -- a
single dense cell hit ~10GB and, combined with everything else running, took
the real desktop down. Process isolation only cleans up AFTER a cell exits;
it does nothing to stop one cell from ballooning memory WHILE it runs. This
sets a hard RLIMIT_AS per worker so pyrosm/networkx hit a clean Python
MemoryError (cell recorded as failed, NaN, retriable) instead of the kernel
ever needing to intervene.

Usage: python3 09b_osm_routing_cell_worker.py <clat> <clon> <cell_deg> <buffer_deg> <sites_csv> <out_csv> <pbf_path>
  sites_csv: Locality_ID,Latitude_WGS84,Longitude_WGS84 for just this cell's sites
  out_csv: where to write Locality_ID,Dist_to_Highway_Routing_m for this cell
"""
import sys, gc, resource
# 2GB was tried first and broke pyrosm's compiled .so loading (mmap-based
# shared-library loading reserves virtual address space well above actual
# RSS). Measured real peak RSS at 0.4deg cells: ~230MB sparse, ~480MB in the
# densest area tested (previously ~10GB at the old 1.5deg cell size) -- 4GB
# leaves generous headroom above real usage while still hard-capping any
# unexpectedly dense cell far below the ~10GB level that crashed the host.
MEM_LIMIT_BYTES = 4 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
import numpy as np, pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
import pyrosm

PAVED_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary",
                  "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}

clat, clon, cell_deg, buffer_deg, sites_csv, out_csv, pbf_path = sys.argv[1:8]
clat, clon, cell_deg, buffer_deg = int(clat), int(clon), float(cell_deg), float(buffer_deg)

group = pd.read_csv(sites_csv)
south = clat * cell_deg - buffer_deg
north = (clat + 1) * cell_deg + buffer_deg
west = clon * cell_deg - buffer_deg
east = (clon + 1) * cell_deg + buffer_deg

rows = []
try:
    osm = pyrosm.OSM(pbf_path, bounding_box=[west, south, east, north])
    result = osm.get_network(network_type="all", nodes=True)
    if result is None or result[0] is None or len(result[0]) == 0:
        print("no road data in this cell")
        for lid in group["Locality_ID"]:
            rows.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": np.nan})
    else:
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
            print("no paved-class roads in this cell")
            for lid in group["Locality_ID"]:
                rows.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": np.nan})
        else:
            Gu = G.to_undirected()
            dist_to_paved = nx.multi_source_dijkstra_path_length(Gu, sources=paved_nodes, weight="length")
            node_ids = np.array(list(G.nodes()))
            node_lat = np.array([G.nodes[n]["y"] for n in node_ids])
            node_lon = np.array([G.nodes[n]["x"] for n in node_ids])
            tree = cKDTree(np.column_stack([node_lon, node_lat]))
            _, idx = tree.query(group[["Longitude_WGS84", "Latitude_WGS84"]].values, k=1)
            nearest_nodes = node_ids[idx]
            for lid, nn in zip(group["Locality_ID"], nearest_nodes):
                rows.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": dist_to_paved.get(nn, np.nan)})
            print(f"cell done: {G.number_of_nodes()} nodes, {len(paved_nodes)} paved nodes")
except Exception as e:
    print(f"CELL FAILED: {type(e).__name__}: {e}")
    for lid in group["Locality_ID"]:
        rows.append({"Locality_ID": lid, "Dist_to_Highway_Routing_m": np.nan})

pd.DataFrame(rows).to_csv(out_csv, index=False)
