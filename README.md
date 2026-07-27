# Enhancing Geoheritage Evaluation in Morocco: Multi-Scale Machine Learning & Spatial Optimization

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-1.1.4-139C5A?style=flat&logo=pandas&logoColor=white)](https://geopandas.org/)
[![Rasterio](https://img.shields.io/badge/Rasterio-1.5.0-008080?style=flat)](https://rasterio.readthedocs.io/)
[![LaTeX](https://img.shields.io/badge/LaTeX-PDFTeX-008080?style=flat&logo=latex&logoColor=white)](https://www.latex-project.org/)
[![Institution](https://img.shields.io/badge/UM6P-GSMI-red?style=flat)](https://www.um6p.ma/)

**Author:** Mohamed EL GORRIM (AI Research Intern)  
**Institution:** Geology and Sustainable Mining Institute (GSMI), Université Mohammed VI Polytechnique (UM6P), Ben Guerir, Morocco  
**Supervisors / Mentors:** Dr. Ismail BEN AMAR, Sanae EL HARCHE, Mohamed EL OUALI  

---

## Executive Overview

This repository contains the complete, production-ready research framework for **multi-scale spatial machine learning, Fuzzy Multicriteria Decision Support System (Fuzzy MCDSS), and loss-free GIS raster feature extraction** applied to geoheritage evaluation and sustainable geotourism management in Morocco.

The project is structured into three modular deliverables:

1. **Phase 1 — National Predictive Accessibility Modeling:**
   - Evaluates continuous Fuzzy MCDSS sigmoid scores ($S_{\text{access}}$) across 309 deduplicated national geosites.
   - Benchmarks classifiers under $0.5^\circ \times 0.5^\circ$ ($\approx 50\text{ km} \times 50\text{ km}$) Spatial Block CV.
   - Achieves a **0.9042 Spatial Weighted F1** score (Accuracy: **91.91%**) with Random Forest.
   - Projects predictions across 484,386 land raster pixels in Morocco.

2. **Phase 2 — Multi-Regional Analytics & Policy Profiling:**
   - Extends localized spatial ML models ($22\text{ km}$ Spatial Block CV) to contrasting pilot regions: **Béni Mellal-Khénifra (BMK)** ($N=55$, 36 mapped to boundary; Best RF Spatial F1 = **0.8585**, Concordance = **85.93%**) and **Tanger-Tétouan-Al Hoceïma (TTAH)** ($N=51$, 47 mapped to boundary; Best RF Spatial F1 = **0.7171**, Concordance = **89.32%**).
   - Computes Brilha (2016) quantitative indices ($V_{\text{sci}}, V_{\text{vuln}}, V_{\text{geo}}$) and unsupervised K-Means ($K=5$) policy management profiles.
   - Incorporates real-world Google Maps transit verification (e.g., Gare de Melloussa station proximity $<750\text{ m}$, Punta Cires 8-min vehicular drive vs 1h05 walk).

3. **Phase 3 — Lossless GIS Raster Feature Extraction CLI Engine:**
   - Production-grade CLI tool (`general_extractor.py`) performing 100% loss-free nearest-neighbor raster queries across 10 physical GIS layers (Elevation, Slope, Ruggedness, Distances to Roads/Rivers/Dams, Geology, Soil, LULC).

---

## Repository Directory Structure

```
geosite_project1/
├── README.md                                  # Root master documentation & usage guide
├── requirements.txt                           # Core dependencies specification
├── .gitignore                                 # Git exclusion rules
├── gis_data/                                  # Raw & physical GeoTIFF rasters
│   ├── physical/                              # 10 projected physical GIS raster layers (Sahara Lambert EPSG:26191)
│   └── boundaries/                            # Regional administrative GeoJSON boundaries
├── references/                                # Excel master datasets & PDF literature
├── models/                                    # Serialized .joblib ML checkpoints
├── livrable/                                  # Modular Phase-Based Deliverables
│   ├── phase1_national_accessibility/         # Phase 1: National Accessibility Modeling
│   │   ├── code/                              # 3 Production Scripts (01_prepare, 02_train, 03_project)
│   │   ├── data/                              # National master CSV datasets
│   │   ├── figures/                           # Publication-grade figures (Seaborn viridis styling)
│   │   ├── models/                            # Model checkpoints
│   │   └── report/                            # LaTeX manuscripts & compiled PDFs (EN & FR)
│   ├── phase2_regional_analytics/             # Phase 2: Regional Analytics & Policy Profiling
│   │   ├── code/                              # 4 Production Scripts (01_prepare, 02_train, 03_project, 04_cluster)
│   │   ├── data/                              # Regional GeoJSON boundaries & indexed CSV datasets
│   │   ├── figures/                           # Annotated regional maps, 3D scatter plots & ground photos
│   │   └── report/                            # LaTeX manuscript & compiled PDF (FR)
│   └── phase3_lossless_extraction_pipeline/   # Phase 3: CLI Raster Feature Extraction Engine
│       ├── code/                              # CLI Extractor & pipeline runner scripts
│       ├── config/                            # YAML raster layer configurations
│       └── report/                            # Technical report
```

---

## Key Results Summary

### Phase 1 — National Accessibility Benchmark (50 km Spatial Block CV)
- **Master Dataset:** 309 deduplicated national geosites.
- **Top Classifier:** **Random Forest Classifier**
  - **Spatial Block CV Weighted F1:** **0.9042**
  - **Spatial Block CV Accuracy:** **91.91%**
  - **Nationwide Land Pixels Projected:** 484,386 pixels (35.3% Easy, 3.8% Moderate, 60.9% Difficult).

### Phase 2 — Regional Model Benchmarks (22 km Spatial Block CV)

| Pilot Region | Total Sites | Mapped to Boundary | Best Spatial Model | Spatial CV F1 | Spatial Accuracy | National Model Concordance |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Béni Mellal-Khénifra (BMK)** | 55 | 36 | **Random Forest (Basic)** | **0.8585** | **87.27%** | **85.93%** (16,829 / 19,585 px) |
| **Tanger-Tétouan-Al Hoceïma (TTAH)** | 51 | 47 | **Random Forest (Basic)** | **0.7171** | **74.51%** | **89.32%** (10,311 / 11,544 px) |

### Phase 2 — K-Means Policy Profiling ($K=5$) & Validation
- **Béni Mellal-Khénifra:** Davies-Bouldin Index local minimum at $K=5$ (**0.9052**), Silhouette = **0.3515**.
- **Tanger-Tétouan-Al Hoceïma:** Mathematical optimum at $K=6$ ($S_{\text{sil}}=\mathbf{0.3371}, DB=\mathbf{0.9120}$); $K=5$ retained for cross-regional policy standardization.

---

## Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/geosite_project1.git
cd geosite_project1

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execute Phase 1 Pipeline (National Accessibility)
```bash
# Step 1: Preprocess national dataset & apply Fuzzy MCDSS engine
python livrable/phase1_national_accessibility/code/01_prepare_fuzzy_mcdss_data.py

# Step 2: Train ML classifiers under 50 km Spatial Block CV & plot Figures 4, 5, 6
python livrable/phase1_national_accessibility/code/02_train_fuzzy_mcdss_model.py

# Step 3: Project predictions across 484,386 land raster pixels in Morocco
python livrable/phase1_national_accessibility/code/03_project_fuzzy_mcdss_map.py
```

### 3. Execute Phase 2 Pipeline (Regional Analytics)
```bash
# Step 1: Prepare regional datasets & compute Brilha (2016) multi-criteria indices
python livrable/phase2_regional_analytics/code/01_prepare_fuzzy_mcdss_phase2.py

# Step 2: Train & benchmark regional models under 22 km Spatial Block CV
python livrable/phase2_regional_analytics/code/02_train_regional_models.py

# Step 3: Project continuous raster maps for BMK & TTAH regions
python livrable/phase2_regional_analytics/code/03_project_regional_maps.py

# Step 4: Perform K-Means clustering (K=5), 3D visualization, and spatial profile mapping
python livrable/phase2_regional_analytics/code/04_unsupervised_clustering.py
```

### 4. Compile LaTeX Manuscripts
```bash
# Compile Phase 1 French Report
pdflatex -output-directory=livrable/phase1_national_accessibility/report livrable/phase1_national_accessibility/report/geosite_internship_report_fr.tex

# Compile Phase 2 French Report
pdflatex -output-directory=livrable/phase2_regional_analytics/report livrable/phase2_regional_analytics/report/geosite_phase2_report_fr.tex
```

---

## Citation & Academic Attribution

If you use this codebase, methodology, or dataset in your academic work, please cite:

```bibtex
@mastersthesis{elgorrim2026geoheritage,
  author       = {Mohamed EL GORRIM},
  title        = {Enhancing Geoheritage Evaluation in Morocco: Multi-Scale Machine Learning and Spatial Optimization},
  school       = {Geology and Sustainable Mining Institute (GSMI), Mohammed VI Polytechnic University (UM6P)},
  address      = {Ben Guerir, Morocco},
  year         = {2026},
  type         = {Internship Research Report}
}
```

---

## License
This project is developed under the research supervision of **GSMI / UM6P**. All rights reserved.
