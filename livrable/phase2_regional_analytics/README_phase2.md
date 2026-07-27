# Phase 2: Regional Predictive Analytics & Multi-Criteria Decision Support Engine (Fuzzy MCDSS)

This repository deliverable documents the localized machine learning pipeline, Brilha (2016) multi-criteria index calculation, spatial block validation ($22\text{ km} \times 22\text{ km}$), and K-Means ($K=5$) unsupervised clustering for two pilot regions in Morocco:
1. **Béni Mellal-Khénifra (BMK, $N=55$ geosites, 36 inside boundary)** — Continental High/Middle Atlas region.
2. **Tanger-Tétouan-Al Hoceïma (TTAH, $N=51$ geosites, 47 inside boundary)** — Rifian coastal/mountainous region.

---

## 1. Localized Machine Learning Benchmark ($22\text{ km}$ Spatial Block CV)

| Pilot Region | Best Spatial Classifier | Standard CV F1 | Spatial Block CV F1 | Spatial Accuracy | Concordance with National Model |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Béni Mellal-Khénifra (BMK)** | **Random Forest (Basic)** | 0.8069 | **0.7235** | **89.09%** | **84.91%** |
| **Tanger-Tétouan-Al Hoceïma (TTAH)** | **HistGradientBoosting (Basic)** | 0.8928 | **0.7510** | **85.56%** | **89.32%** |

---

## 2. Brilha (2016) Multi-Criteria Indices ($1.0$ to $5.0$)

For territorial planning and conservation strategy, three multi-criteria indices are computed:
- **Scientific Value ($V_{\text{sci}}$):** Combines geological interest weight and regional lithological rarity.
- **Vulnerability Index ($V_{\text{vuln}}$):** Merges slope erosion susceptibility ($S_{\text{erosion}}$), soil sensitivity ($S_{\text{soil}}$), and anthropogenic pressure ($P_{\text{anthro}}$).
- **Geotouristic Potential ($V_{\text{geo}}$):** Merges visual aesthetic quality and physical access safety.

---

## 3. Unsupervised K-Means Clustering ($K=5$)

Validated via Elbow inertia, Silhouette score, and Davies-Bouldin index:
1. **Préservation Absolue (Sanctuaire):** High scientific value, extreme isolation, high vulnerability.
2. **Préservation Passive (Accès difficile):** Isolated sites requiring access control.
3. **Conservation Active (Sensibilité physique):** Accessible sites vulnerable to physical erosion.
4. **Promotion Éducative (Grand Public):** Highly accessible, low vulnerability, suitable for school trips.
5. **Promotion Géotouristique Locale:** Highly accessible valley/coastal sites with strong eco-tourism potential.

---

## 4. Execution Pipeline

To run the complete 4-step Phase 2 pipeline sequentially:

```bash
# Step 1: Preprocess datasets, apply Fuzzy MCDSS engine & compute Brilha (2016) indices
python phase2_regional_analytics/code/01_prepare_fuzzy_mcdss_phase2.py

# Step 2: Train and benchmark regional ML models under 22 km Spatial Block CV
python phase2_regional_analytics/code/02_train_regional_models.py

# Step 3: Project continuous raster predictions across regional land grids
python phase2_regional_analytics/code/03_project_regional_maps.py

# Step 4: Perform K-Means clustering (K=5), 3D visualization, and spatial profile mapping
python phase2_regional_analytics/code/04_unsupervised_clustering.py
```
