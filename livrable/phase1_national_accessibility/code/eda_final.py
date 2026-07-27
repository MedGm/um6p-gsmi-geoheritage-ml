"""
Comprehensive EDA on geosites_physical_features.csv (post-fix dataset).
Generates 6 figure panels saved as artifacts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch

# ── Style ──
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

workspace = "/home/medgm/um6p-intern/geosite_project1"
artifact_dir = "/home/medgm/.gemini/antigravity/brain/81393a77-68e6-4d4a-84df-6232b7caa664"
df = pd.read_csv(f"{workspace}/scratch/geosites_physical_features.csv")

# Class label lookups
lulc_names = {
    1:'Artificial areas', 2:'Closed broadleaved deciduous', 3:'Irrigated croplands',
    4:'Mosaic Vegetation/Croplands', 5:'Bare areas', 6:'Closed broadleaved flooded',
    7:'Closed needleleaved evergreen', 8:'Closed to open shrubland',
    9:'Mixed broadleaved/needleleaved', 10:'Mosaic Croplands/Vegetation',
    11:'Mosaic Forest-Shrubland', 12:'Mosaic Grassland/Forest', 13:'Rainfed croplands',
    14:'Semi-deciduous forest', 15:'Sparse vegetation', 16:'Water bodies'
}
soil_names = {
    1:'A Sesquioxydes', 2:'Brunifiés', 3:'Calcimagnésiques', 4:'Fersiallitiques',
    5:'Isohumiques', 6:'Min. Bruts Apport', 7:'Min. Bruts Érosion',
    8:'Peu Évolués Apport', 9:'Peu Évolués Érosion', 10:'Vertisols',
    11:'Vertisols/Isohumique'
}

print(f"Dataset: {len(df)} geosites, {len(df.columns)} columns")
print(f"Complete rows (no NaN in physical features): "
      f"{df.dropna(subset=['Elevation_m','Slope_deg','Ruggedness','Dist_to_Dam_m','Dist_to_River_m','LULC_Class','Soil_Class','Geology_Class']).shape[0]}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Overview — Missing data + Target distribution
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1a. Missing data heatmap
phys_cols = ['Elevation_m','Slope_deg','Ruggedness','Dist_to_Dam_m','Dist_to_River_m',
             'LULC_Class','Soil_Class','Geology_Class','Domain_Geosite_Count','Region_Geosite_Count']
missing_pct = df[phys_cols].isnull().mean() * 100
colors = ['#e74c3c' if v > 20 else '#f39c12' if v > 10 else '#2ecc71' for v in missing_pct]
bars = axes[0].barh(range(len(phys_cols)), missing_pct, color=colors, edgecolor='white', linewidth=0.5)
axes[0].set_yticks(range(len(phys_cols)))
axes[0].set_yticklabels([c.replace('_', ' ') for c in phys_cols], fontsize=8)
axes[0].set_xlabel('% Missing')
axes[0].set_title('Data Completeness', fontweight='bold')
for i, v in enumerate(missing_pct):
    axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=7)
axes[0].set_xlim(0, max(missing_pct) * 1.3)
axes[0].invert_yaxis()

# 1b. Geosite Type distribution
type_counts = df['Geosite_Type'].dropna().value_counts().head(10)
palette = sns.color_palette("Set2", n_colors=len(type_counts))
axes[1].barh(range(len(type_counts)), type_counts.values, color=palette, edgecolor='white')
axes[1].set_yticks(range(len(type_counts)))
axes[1].set_yticklabels(type_counts.index, fontsize=8)
axes[1].set_xlabel('Count')
axes[1].set_title('Geosite Type Distribution (Top 10)', fontweight='bold')
axes[1].invert_yaxis()

# 1c. Geological Domain distribution
domain_counts = df['Geological_Domain'].str.strip().value_counts()
axes[2].barh(range(len(domain_counts)), domain_counts.values, color=sns.color_palette("husl", len(domain_counts)), edgecolor='white')
axes[2].set_yticks(range(len(domain_counts)))
axes[2].set_yticklabels(domain_counts.index, fontsize=7)
axes[2].set_xlabel('Count')
axes[2].set_title('Geological Domain Distribution', fontweight='bold')
axes[2].invert_yaxis()

plt.tight_layout()
fig.savefig(f"{artifact_dir}/eda_01_overview.png", bbox_inches='tight')
plt.close()
print("Saved eda_01_overview.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Continuous feature distributions
# ═══════════════════════════════════════════════════════════════
cont_features = ['Elevation_m', 'Slope_deg', 'Ruggedness', 'Dist_to_Dam_m', 'Dist_to_River_m']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feat in enumerate(cont_features):
    data = df[feat].dropna()
    axes[i].hist(data, bins=40, color=sns.color_palette("viridis", 5)[i], edgecolor='white', alpha=0.85)
    axes[i].axvline(data.median(), color='red', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.0f}')
    axes[i].axvline(data.mean(), color='orange', linestyle=':', linewidth=1.5, label=f'Mean: {data.mean():.0f}')
    axes[i].set_title(feat.replace('_', ' '), fontweight='bold')
    axes[i].set_ylabel('Count')
    axes[i].legend(fontsize=7)

# Box plots in the 6th panel
axes[5].boxplot([df[f].dropna() for f in cont_features], 
                tick_labels=[f.replace('_m','').replace('_deg','') for f in cont_features],
                patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.6))
axes[5].set_title('Feature Ranges (Box Plot)', fontweight='bold')
axes[5].tick_params(axis='x', rotation=30, labelsize=7)

plt.suptitle('Continuous Feature Distributions (n=375 geosites)', fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(f"{artifact_dir}/eda_02_continuous.png", bbox_inches='tight')
plt.close()
print("Saved eda_02_continuous.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Categorical feature distributions (LULC, Soil)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LULC
lulc_counts = df['LULC_Class'].dropna().astype(int).map(lulc_names).value_counts()
axes[0].barh(range(len(lulc_counts)), lulc_counts.values, 
             color=sns.color_palette("Spectral", len(lulc_counts)), edgecolor='white')
axes[0].set_yticks(range(len(lulc_counts)))
axes[0].set_yticklabels(lulc_counts.index, fontsize=7)
axes[0].set_xlabel('Number of Geosites')
axes[0].set_title('Land Use / Land Cover at Geosite Locations', fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(lulc_counts.values):
    axes[0].text(v + 0.3, i, str(v), va='center', fontsize=7)

# Soil
soil_counts = df['Soil_Class'].dropna().astype(int).map(soil_names).value_counts()
axes[1].barh(range(len(soil_counts)), soil_counts.values,
             color=sns.color_palette("gist_earth", len(soil_counts)), edgecolor='white')
axes[1].set_yticks(range(len(soil_counts)))
axes[1].set_yticklabels(soil_counts.index, fontsize=8)
axes[1].set_xlabel('Number of Geosites')
axes[1].set_title('Soil Type at Geosite Locations', fontweight='bold')
axes[1].invert_yaxis()
for i, v in enumerate(soil_counts.values):
    axes[1].text(v + 0.3, i, str(v), va='center', fontsize=7)

plt.tight_layout()
fig.savefig(f"{artifact_dir}/eda_03_categorical.png", bbox_inches='tight')
plt.close()
print("Saved eda_03_categorical.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Correlation matrix
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = ['Elevation_m','Slope_deg','Ruggedness','Dist_to_Dam_m','Dist_to_River_m',
             'LULC_Class','Soil_Class','Geology_Class','Domain_Geosite_Count','Region_Geosite_Count',
             'Latitude_WGS84','Longitude_WGS84']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            xticklabels=[c.replace('_',' ') for c in corr_cols],
            yticklabels=[c.replace('_',' ') for c in corr_cols],
            annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
ax.tick_params(axis='both', labelsize=7)

plt.tight_layout()
fig.savefig(f"{artifact_dir}/eda_04_correlation.png", bbox_inches='tight')
plt.close()
print("Saved eda_04_correlation.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Spatial distribution map
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 5a. Colored by elevation
valid = df.dropna(subset=['Elevation_m'])
sc = axes[0].scatter(valid['Longitude_WGS84'], valid['Latitude_WGS84'], 
                     c=valid['Elevation_m'], cmap='terrain', s=15, alpha=0.7, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=axes[0], label='Elevation (m)', shrink=0.8)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('Geosites Colored by Elevation', fontweight='bold')
axes[0].set_aspect('equal')

# 5b. Colored by geological domain
domains = df['Geological_Domain'].str.strip().unique()
domain_colors = {d: c for d, c in zip(sorted(domains), sns.color_palette("husl", len(domains)))}
for domain in sorted(domains):
    subset = df[df['Geological_Domain'].str.strip() == domain]
    axes[1].scatter(subset['Longitude_WGS84'], subset['Latitude_WGS84'], 
                    c=[domain_colors[domain]], s=15, alpha=0.7, label=domain, edgecolors='k', linewidths=0.3)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].set_title('Geosites Colored by Geological Domain', fontweight='bold')
axes[1].legend(fontsize=6, loc='lower left', ncol=2, framealpha=0.8)
axes[1].set_aspect('equal')

plt.tight_layout()
fig.savefig(f"{artifact_dir}/eda_05_spatial.png", bbox_inches='tight')
plt.close()
print("Saved eda_05_spatial.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Key bivariate relationships
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# 6a. Elevation vs Slope
valid = df.dropna(subset=['Elevation_m', 'Slope_deg'])
axes[0,0].scatter(valid['Elevation_m'], valid['Slope_deg'], s=12, alpha=0.5, c='#3498db', edgecolors='none')
axes[0,0].set_xlabel('Elevation (m)')
axes[0,0].set_ylabel('Slope (degrees)')
axes[0,0].set_title('Elevation vs Slope', fontweight='bold')

# 6b. Distance to Dam vs Distance to River
valid = df.dropna(subset=['Dist_to_Dam_m', 'Dist_to_River_m'])
axes[0,1].scatter(valid['Dist_to_Dam_m']/1000, valid['Dist_to_River_m']/1000, 
                  s=12, alpha=0.5, c='#e74c3c', edgecolors='none')
axes[0,1].set_xlabel('Distance to Dam (km)')
axes[0,1].set_ylabel('Distance to River (km)')
axes[0,1].set_title('Dam Distance vs River Distance', fontweight='bold')

# 6c. Elevation by Geological Domain (violin)
valid = df.dropna(subset=['Elevation_m'])
top_domains = valid['Geological_Domain'].str.strip().value_counts().head(6).index
domain_data = [valid[valid['Geological_Domain'].str.strip() == d]['Elevation_m'].values for d in top_domains]
parts = axes[1,0].violinplot(domain_data, positions=range(len(top_domains)), showmeans=True, showmedians=True)
axes[1,0].set_xticks(range(len(top_domains)))
axes[1,0].set_xticklabels(top_domains, fontsize=7, rotation=25, ha='right')
axes[1,0].set_ylabel('Elevation (m)')
axes[1,0].set_title('Elevation by Geological Domain', fontweight='bold')

# 6d. LULC by Geological Domain (stacked bar)
valid = df.dropna(subset=['LULC_Class'])
valid['LULC_Name'] = valid['LULC_Class'].astype(int).map(lulc_names)
valid['Domain_Clean'] = valid['Geological_Domain'].str.strip()
ct = pd.crosstab(valid['Domain_Clean'], valid['LULC_Name'], normalize='index')
top_lulc = ct.sum().nlargest(6).index
ct_top = ct[top_lulc]
ct_top.plot(kind='barh', stacked=True, ax=axes[1,1], colormap='Set3', edgecolor='white', linewidth=0.3)
axes[1,1].set_xlabel('Proportion')
axes[1,1].set_title('Land Cover Composition by Domain', fontweight='bold')
axes[1,1].legend(fontsize=6, loc='lower right', framealpha=0.8)
axes[1,1].tick_params(axis='y', labelsize=7)

plt.suptitle('Bivariate Feature Relationships', fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(f"{artifact_dir}/eda_06_bivariate.png", bbox_inches='tight')
plt.close()
print("Saved eda_06_bivariate.png")

# ═══════════════════════════════════════════════════════════════
# Print summary statistics
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(df[['Elevation_m','Slope_deg','Ruggedness','Dist_to_Dam_m','Dist_to_River_m']].describe().round(1).to_string())

print("\n\nLULC CLASS DISTRIBUTION:")
for cls, name in sorted(lulc_names.items()):
    count = (df['LULC_Class'] == cls).sum()
    if count > 0:
        print(f"  {cls:2d}. {name:40s}: {count:3d} geosites")

print("\nSOIL CLASS DISTRIBUTION:")
for cls, name in sorted(soil_names.items()):
    count = (df['Soil_Class'] == cls).sum()
    if count > 0:
        print(f"  {cls:2d}. {name:30s}: {count:3d} geosites")

print("\nGEOLOGY: Top 10 formation classes:")
print(df['Geology_Class'].dropna().astype(int).value_counts().head(10).to_string())

print(f"\n\nAll 6 EDA figures saved to {artifact_dir}/")
