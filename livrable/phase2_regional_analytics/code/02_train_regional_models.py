import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from xgboost import XGBClassifier

PHASE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PHASE2_DIR, "data")
FIGURES_DIR = os.path.join(PHASE2_DIR, "figures")
MODELS_DIR = os.path.abspath(os.path.join(PHASE2_DIR, "..", "..", "models"))
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    'Elevation_m', 'Slope_deg', 'Ruggedness', 'Dist_to_Dam_m', 'Dist_to_River_m',
    'LULC_Class', 'Soil_Class', 'Geology_Class', 'Dist_to_Highway_m', 'Dist_to_Piste_m'
]

LABEL_MAP = {'Easy': 0, 'Moderate': 1, 'Difficult': 2}

def get_spatial_blocks(lats, lons, block_size=0.2):
    lat_b = (np.array(lats) / block_size).astype(int)
    lon_b = (np.array(lons) / block_size).astype(int)
    return np.array([f"{la}_{lo}" for la, lo in zip(lat_b, lon_b)])

def benchmark_region(csv_path, region_name, fig_name, model_out_name):
    print(f"\n" + "="*80)
    print(f"   BENCHMARKING REGIONAL MODELS FOR {region_name} (22 km SPATIAL BLOCK CV)")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = df['Accessibility'].map(LABEL_MAP).values
    lats = df['Latitude_WGS84'].values
    lons = df['Longitude_WGS84'].values
    groups = get_spatial_blocks(lats, lons, block_size=0.2)

    print(f"Master dataset size: {len(df)} geosites across {len(np.unique(groups))} spatial blocks.")

    models = {
        "Random Forest (Basic)": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
        "XGBoost (Basic)": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='mlogloss'),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, max_iter=150, random_state=42)
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5 if len(np.unique(groups)) >= 5 else len(np.unique(groups)))

    results = []
    best_f1 = -1.0
    best_model_name = ""
    best_model_obj = None

    for name, model in models.items():
        # Standard Stratified CV
        y_true_std, y_pred_std = [], []
        for tr_idx, te_idx in skf.split(X, y):
            model.fit(X.iloc[tr_idx], y[tr_idx])
            y_pred_std.extend(model.predict(X.iloc[te_idx]))
            y_true_std.extend(y[te_idx])
        acc_std = accuracy_score(y_true_std, y_pred_std)
        f1_std = f1_score(y_true_std, y_pred_std, average='weighted')

        # Spatial Block CV
        y_true_sp, y_pred_sp = [], []
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            model.fit(X.iloc[tr_idx], y[tr_idx])
            y_pred_sp.extend(model.predict(X.iloc[te_idx]))
            y_true_sp.extend(y[te_idx])
        acc_sp = accuracy_score(y_true_sp, y_pred_sp)
        f1_sp = f1_score(y_true_sp, y_pred_sp, average='weighted')

        results.append({
            "Model": name,
            "Standard Acc": acc_std,
            "Standard F1": f1_std,
            "Spatial Acc": acc_sp,
            "Spatial F1": f1_sp,
            "Leakage Gap": abs(f1_std - f1_sp)
        })

        print(f"Model: {name:25s} | Standard F1: {f1_std:.4f} | Spatial Acc: {acc_sp:.4f} | Spatial Weighted F1: {f1_sp:.4f}")

        if f1_sp > best_f1:
            best_f1 = f1_sp
            best_model_name = name
            best_model_obj = model

    print(f"\nBest spatial model for {region_name}: {best_model_name} (Spatial F1 = {best_f1:.4f})")
    best_model_obj.fit(X, y)

    # Save model artifact
    model_path = os.path.join(MODELS_DIR, model_out_name)
    joblib.dump(best_model_obj, model_path)
    print(f"Saved model artifact -> {model_path}")

    # Plot Model Comparison Bar Chart
    df_res = pd.DataFrame(results)
    plot_data = []
    for _, row in df_res.iterrows():
        plot_data.append({"Model": row["Model"], "Validation Scheme": "Standard Stratified", "F1 Score": row["Standard F1"]})
        plot_data.append({"Model": row["Model"], "Validation Scheme": "Spatial Block CV (22km)", "F1 Score": row["Spatial F1"]})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_plot, x="Model", y="F1 Score", hue="Validation Scheme", palette=["#2ecc71", "#3498db"])
    plt.title(f"Phase 2 — {region_name} Accessibility Model Benchmark", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(0.4, 1.02)
    plt.ylabel("Weighted Macro F1 Score", fontsize=10)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3), textcoords='offset points')

    plt.legend(loc="lower right")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Figure -> {fig_path}")

def main():
    benchmark_region(os.path.join(DATA_DIR, "geosites_bmk_indexed.csv"), "Béni Mellal-Khénifra (BMK)", "regional_model_comparison.png", "bmk_regional_model.joblib")
    benchmark_region(os.path.join(DATA_DIR, "geosites_ttah_indexed.csv"), "Tanger-Tétouan-Al Hoceïma (TTAH)", "ttah_model_comparison.png", "ttah_regional_model.joblib")

if __name__ == "__main__":
    main()
