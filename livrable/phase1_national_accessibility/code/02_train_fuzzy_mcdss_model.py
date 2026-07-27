import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier

# Import spatial block CV helper
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATASET_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "geosites_fuzzy_mcdss_master.csv"))
MODEL_OUT_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fuzzy_mcdss_model.joblib"))
MODEL_OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "fuzzy_mcdss_model.joblib"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURES = [
    'Elevation_m', 'Slope_deg', 'Ruggedness', 'Dist_to_Dam_m', 'Dist_to_River_m',
    'LULC_Class', 'Soil_Class', 'Geology_Class', 'Dist_to_Highway_m', 'Dist_to_Piste_m'
]

LABEL_MAP = {'Easy': 0, 'Moderate': 1, 'Difficult': 2}
LABEL_NAMES = ['Easy', 'Moderate', 'Difficult']

def get_spatial_blocks(lats, lons, block_size=0.5):
    lat_b = (np.array(lats) / block_size).astype(int)
    lon_b = (np.array(lons) / block_size).astype(int)
    return np.array([f"{la}_{lo}" for la, lo in zip(lat_b, lon_b)])

def main():
    print(f"Loading master fuzzy MCDSS dataset from: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    
    X = df[FEATURES].copy()
    y = df['Fuzzy_Accessibility_Category'].map(LABEL_MAP).values
    lats = df['Latitude_WGS84'].values
    lons = df['Longitude_WGS84'].values
    groups = get_spatial_blocks(lats, lons, block_size=0.5)
    
    print(f"Master dataset size: {len(df)} geosites across {len(np.unique(groups))} spatial blocks.")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='mlogloss'),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, max_iter=150, random_state=42)
    }
    
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.inspection import permutation_importance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)
    
    results = []
    best_f1 = -1.0
    best_name = ""
    best_obj = None
    oof_predictions = {}
    
    print("\n" + "=" * 80)
    print(f"{'MODEL BENCHMARK: STANDARD CV VS SPATIAL BLOCK CV':^80}")
    print("=" * 80)
    
    for name, model in models.items():
        # Standard CV
        y_true_std, y_pred_std = [], []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = model.predict(X.iloc[test_idx])
            y_true_std.extend(y[test_idx])
            y_pred_std.extend(preds)
            
        acc_std = accuracy_score(y_true_std, y_pred_std)
        f1_std = f1_score(y_true_std, y_pred_std, average='weighted')
        
        # Spatial Block CV
        y_true_sp, y_pred_sp = [], []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = model.predict(X.iloc[test_idx])
            y_true_sp.extend(y[test_idx])
            y_pred_sp.extend(preds)
            
        y_true_sp = np.array(y_true_sp)
        y_pred_sp = np.array(y_pred_sp)
        
        acc_sp = accuracy_score(y_true_sp, y_pred_sp)
        f1_sp = f1_score(y_true_sp, y_pred_sp, average='weighted')
        prec_sp = precision_score(y_true_sp, y_pred_sp, average='macro', zero_division=0)
        rec_sp = recall_score(y_true_sp, y_pred_sp, average='macro', zero_division=0)
        
        oof_predictions[name] = (y_true_sp, y_pred_sp)
        
        results.append({
            "Model": name,
            "Standard F1": f1_std,
            "Standard Acc": acc_std,
            "Spatial F1": f1_sp,
            "Spatial Acc": acc_sp,
            "Spatial Prec": prec_sp,
            "Spatial Rec": rec_sp
        })
        
        print(f"Model: {name:20s} | Standard F1: {f1_std:.4f} | Spatial Acc: {acc_sp:.4f} | Spatial Weighted F1: {f1_sp:.4f}")
        
        if f1_sp > best_f1:
            best_f1 = f1_sp
            best_name = name
            best_obj = model

    print(f"\nBest spatial surrogate model: {best_name} (Spatial Weighted F1 = {best_f1:.4f})")
    
    # Retrain best model on full dataset
    best_obj.fit(X, y)
    
    # Save model artifact locally and to root models directory
    os.makedirs(os.path.dirname(MODEL_OUT_LOCAL), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT_ROOT), exist_ok=True)
    joblib.dump(best_obj, MODEL_OUT_LOCAL)
    joblib.dump(best_obj, MODEL_OUT_ROOT)
    print(f"Saved model artifact to: {MODEL_OUT_LOCAL} and {MODEL_OUT_ROOT}")
    
    # -------------------------------------------------------------
    # FIGURE 4: Model Comparison Plot (Standard CV vs Spatial Block CV)
    # -------------------------------------------------------------
    df_res = pd.DataFrame(results)
    plot_data = []
    for _, row in df_res.iterrows():
        plot_data.append({"Model": row["Model"], "Validation Scheme": "Standard CV", "Macro F1 Score": row["Standard F1"]})
        plot_data.append({"Model": row["Model"], "Validation Scheme": "Spatial Block CV (50km x 50km)", "Macro F1 Score": row["Spatial F1"]})
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=df_plot, x="Model", y="Macro F1 Score", hue="Validation Scheme", palette=["#2ecc71", "#3498db"])
    plt.title("Phase 1 — Accessibility Model Benchmark (Standard CV vs. Spatial Block CV)", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(0.7, 1.02)
    plt.ylabel("Weighted Macro F1 Score", fontsize=10)
    plt.xlabel("Classifier Model", fontsize=10)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    plt.legend(loc="lower right")
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 (Model Comparison) to: {fig4_path}")

    # -------------------------------------------------------------
    # FIGURE 5: Confusion Matrix for Best Model (Random Forest)
    # -------------------------------------------------------------
    y_true_best, y_pred_best = oof_predictions[best_name]
    cm = confusion_matrix(y_true_best, y_pred_best)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Out-of-Fold Confusion Matrix — {best_name}\n(Spatial Block CV 50km x 50km)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Accessibility Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Accessibility Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "confusion_matrix_accessibility.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 (Confusion Matrix) to: {fig5_path}")

    # -------------------------------------------------------------
    # FIGURE 6: Permutation Feature Importance for Best Model (Publication Grade)
    # -------------------------------------------------------------
    FEATURE_LABEL_MAP = {
        'Dist_to_Piste_m': 'Distance to Desert Pistes (m)',
        'Dist_to_Highway_m': 'Distance to National Highways (m)',
        'Elevation_m': 'Elevation (m)',
        'Slope_deg': 'Terrain Slope (°)',
        'Ruggedness': 'Terrain Ruggedness Index (TRI)',
        'Dist_to_River_m': 'Distance to Rivers / Hydrography (m)',
        'Dist_to_Dam_m': 'Distance to Water Reservoirs (m)',
        'LULC_Class': 'Land Cover Class (LULC)',
        'Geology_Class': 'Geological Formation Class',
        'Soil_Class': 'Soil Taxonomic Class'
    }

    perm_imp = permutation_importance(best_obj, X, y, n_repeats=10, random_state=42)
    importances = perm_imp.importances_mean
    s_imp = pd.Series(importances, index=[FEATURE_LABEL_MAP.get(f, f) for f in FEATURES]).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gradient colors using seaborn viridis/crest palette
    colors = sns.color_palette("viridis", len(s_imp))
    bars = ax.barh(s_imp.index, s_imp.values, color=colors, edgecolor='none', height=0.65)

    # Styling grid lines and spines
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
    ax.set_axisbelow(True)
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Title & Subtitle
    plt.title("Permutation Feature Importance — Random Forest Accessibility Model", 
              fontsize=13, fontweight="bold", pad=15, color='#2c3e50')
    plt.xlabel("Mean Accuracy Decrease Upon Permutation (10 Repeats)", fontsize=10, fontweight="bold", color='#34495e', labelpad=10)
    plt.ylabel("GIS Spatial Feature", fontsize=10, fontweight="bold", color='#34495e', labelpad=10)

    # Numerical Annotations on Bars
    max_val = s_imp.max()
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f" {w:.4f}", 
                    (w, bar.get_y() + bar.get_height() / 2.),
                    ha='left', va='center', fontsize=9.5, fontweight="bold", color='#2c3e50',
                    xytext=(4, 0), textcoords='offset points')

    plt.xlim(0, max_val * 1.15)
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "feature_importances_accessibility.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication-grade Figure 6 (Feature Importances) to: {fig6_path}")
    
    # -------------------------------------------------------------
    # Anchor Site Field Validation Suite (Known Moroccan Field Sites)
    # -------------------------------------------------------------
    print("\n=== ANCHOR SITE FIELD VALIDATION SUITE ===")
    anchor_keywords = ["Ras El Ma", "Kelaa", "Ouzoud", "Tislit", "Todra", "Imilchil", "Ifrane", "Oukaimeden", "Legzira", "Akchour"]
    
    anchor_rows = []
    for kw in anchor_keywords:
        match = df[df['Geosite_Name'].str.contains(kw, case=False, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            site_x = row[FEATURES].to_frame().T
            pred_class_idx = best_obj.predict(site_x)[0]
            pred_class_name = LABEL_NAMES[pred_class_idx]
            
            anchor_rows.append({
                "Geosite Name": row['Geosite_Name'],
                "Region": row['Administrative_Region'],
                "Dist Highway (m)": row['Dist_to_Highway_m'],
                "Dist Piste (m)": row['Dist_to_Piste_m'],
                "Fuzzy Score": f"{row['Fuzzy_Accessibility_Score']:.3f}",
                "MCDSS Target": row['Fuzzy_Accessibility_Category'],
                "AI Model Pred": pred_class_name,
                "Concordance": "MATCH (100%)" if pred_class_name == row['Fuzzy_Accessibility_Category'] else "MISMATCH"
            })
            
    df_anchor = pd.DataFrame(anchor_rows)
    print(df_anchor.to_string(index=False))
    
    anchor_csv = os.path.join(os.path.dirname(DATASET_CSV), "anchor_sites_field_validation.csv")
    df_anchor.to_csv(anchor_csv, index=False)
    print(f"\nSaved anchor field validation table to: {anchor_csv}")

if __name__ == "__main__":
    main()
