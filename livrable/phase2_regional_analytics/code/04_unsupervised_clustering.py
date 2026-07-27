import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

PHASE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PHASE2_DIR, "data")
FIGURES_DIR = os.path.join(PHASE2_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLUSTER_FEATURES = ['V_sci', 'V_vuln', 'Fuzzy_Accessibility_Score']
CLUSTER_NAMES = {
    0: 'Préservation Absolue (Sanctuaire)',
    1: 'Préservation Passive (Accès difficile)',
    2: 'Conservation Active (Sensibilité physique)',
    3: 'Promotion Éducative (Grand Public)',
    4: 'Promotion Géotouristique Locale'
}
PALETTE = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

def run_clustering_pipeline(region_name, csv_path, k_just_fig, clust_3d_fig, clust_spat_fig):
    print(f"\n" + "="*80)
    print(f"   K-MEANS CLUSTERING & PROFILE CARACTERIZATION FOR {region_name} (K=5)")
    print("="*80)

    df = pd.read_csv(csv_path)
    X = df[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. K Selection Justification Plot (Elbow, Silhouette, Davies-Bouldin)
    k_range = range(2, 9)
    inertias, sil_scores, db_scores = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=7)
    ax1.set_title("Méthode Elbow (Inertie Intra-Classe)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Nombre de Clusters K", fontsize=10)
    ax1.set_ylabel("Inertie Intra-Classe", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(k_range, sil_scores, 'go-', linewidth=2, markersize=7)
    ax2.set_title("Score Silhouette (Max est mieux)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Nombre de Clusters K", fontsize=10)
    ax2.set_ylabel("Score Silhouette", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax3.plot(k_range, db_scores, 'ro-', linewidth=2, markersize=7)
    ax3.set_title("Index Davies-Bouldin (Min est mieux)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Nombre de Clusters K", fontsize=10)
    ax3.set_ylabel("Index Davies-Bouldin", fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f"Phase 2 — Validation du Partitionnement K-Means ({region_name})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, k_just_fig)
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved K Justification Figure -> {fig1_path}")

    # 2. Fit K=5 Final Model
    km_final = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster_ID'] = km_final.fit_predict(X_scaled)
    df['Management_Profile'] = df['Cluster_ID'].map(CLUSTER_NAMES)

    # Save updated dataframe with cluster assignments
    df.to_csv(csv_path, index=False)

    # 3. 3D Cluster Scatter Plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for cid in range(5):
        cdata = df[df['Cluster_ID'] == cid]
        ax.scatter(cdata['V_sci'], cdata['V_vuln'], cdata['Fuzzy_Accessibility_Score'],
                   c=PALETTE[cid], label=CLUSTER_NAMES[cid], s=50, edgecolors='k', alpha=0.85)

    ax.set_xlabel('Valeur Scientifique (V_sci)', fontsize=10, labelpad=8)
    ax.set_ylabel('Index Vulnérabilité (V_vuln)', fontsize=10, labelpad=8)
    ax.set_zlabel('Score Flou Accessibilité (S_access)', fontsize=10, labelpad=8)
    plt.title(f"Clustering 3D des Géosites (K=5) — {region_name}", fontsize=12, fontweight="bold", pad=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8.5)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, clust_3d_fig)
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D Clusters Figure -> {fig2_path}")

    # 4. Spatial Map of Clusters with Administrative Boundary & Enlarged Markers
    import geopandas as gpd
    
    geojson_filename = "beni_mellal_khenifra_boundary.geojson" if "Béni" in region_name else "tanger_tetouan_al_hoceima_boundary.geojson"
    boundary_path = os.path.join(DATA_DIR, geojson_filename)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if os.path.exists(boundary_path):
        gdf = gpd.read_file(boundary_path)
        gdf.plot(ax=ax, facecolor='#f8f9fa', edgecolor='#2c3e50', linewidth=1.8, alpha=0.85, zorder=1)
        bounds = gdf.total_bounds
        margin_x = (bounds[2] - bounds[0]) * 0.05
        margin_y = (bounds[3] - bounds[1]) * 0.05
        ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
        ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)
    
    df_inside = df[df['Is_Inside_Boundary'] == True] if 'Is_Inside_Boundary' in df.columns else df
    
    for cid in range(5):
        cdata = df_inside[df_inside['Cluster_ID'] == cid]
        ax.scatter(cdata['Longitude_WGS84'], cdata['Latitude_WGS84'],
                   c=PALETTE[cid], label=CLUSTER_NAMES[cid], s=110, edgecolors='black', linewidth=1.1, zorder=3, alpha=0.95)

    plt.title(f"Répartition Géographique des Profils de Gestion K-Means (K=5)\n{region_name} (N={len(df_inside)} sites ancrés au territoire)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Longitude WGS84", fontsize=10)
    plt.ylabel("Latitude WGS84", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.legend(loc='lower right', fontsize=8.5, framealpha=0.95, edgecolor='gray')
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, clust_spat_fig)
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Spatial Clusters Figure -> {fig3_path}\n")

def main():
    run_clustering_pipeline("Béni Mellal-Khénifra", os.path.join(DATA_DIR, "geosites_bmk_indexed.csv"),
                            "kmeans_k_justification.png", "geosites_3d_clustering.png", "geosites_spatial_clusters.png")

    run_clustering_pipeline("Tanger-Tétouan-Al Hoceïma", os.path.join(DATA_DIR, "geosites_ttah_indexed.csv"),
                            "ttah_kmeans_k_justification.png", "ttah_geosites_3d_clustering.png", "ttah_geosites_spatial_clusters.png")

if __name__ == "__main__":
    main()
