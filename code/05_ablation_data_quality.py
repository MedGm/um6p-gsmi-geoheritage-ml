"""
Ablation: does raster-imputation quality actually affect model performance?

Trains the same model benchmark on three variants of the same 375-locality dataset:
  Full            (375) - everything, no filtering
  Quality subset  (362) - excludes the 13 rows where ALL 8 rasters returned NoData
  High-confidence (161) - only rows where ZERO raster cells needed imputation

If Spatial Block CV performance is close across all three, imputation isn't materially
hurting the model and dropping rows for "cleanliness" would just waste data. If the
high-confidence subset performs meaningfully better, that's evidence the imputed rows
are adding noise worth removing.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "training")
FEATURES_CSV = os.path.join(DATA_DIR, "geosites_features.csv")
LABELS_CSV = os.path.join(DATA_DIR, "geosites_labels_draft.csv")

FEATURES = [
    "Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Dam_m", "Dist_to_River_m",
    "LULC_Class", "Soil_Class", "Geology_Class", "Dist_to_Highway_m",
]


def get_spatial_blocks(lats, lons, block_size=0.5):
    lat_b = (np.array(lats) / block_size).astype(int)
    lon_b = (np.array(lons) / block_size).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(lat_b, lon_b)])


def benchmark(df, variant_name):
    X = df[FEATURES].copy()
    label_map = {lbl: i for i, lbl in enumerate(sorted(df["Draft_Accessibility_Class"].unique()))}
    y = df["Draft_Accessibility_Class"].map(label_map).values
    groups = get_spatial_blocks(df["Latitude_WGS84"].values, df["Longitude_WGS84"].values)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="mlogloss"),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, max_iter=150, random_state=42),
    }

    n_splits = min(5, pd.Series(y).value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))

    rows = []
    for name, model in models.items():
        y_true_std, y_pred_std = [], []
        for tr, te in skf.split(X, y):
            model.fit(X.iloc[tr], y[tr])
            y_true_std += list(y[te]); y_pred_std += list(model.predict(X.iloc[te]))
        f1_std = f1_score(y_true_std, y_pred_std, average="weighted")
        acc_std = accuracy_score(y_true_std, y_pred_std)

        y_true_sp, y_pred_sp = [], []
        for tr, te in gkf.split(X, y, groups=groups):
            model.fit(X.iloc[tr], y[tr])
            y_true_sp += list(y[te]); y_pred_sp += list(model.predict(X.iloc[te]))
        f1_sp = f1_score(y_true_sp, y_pred_sp, average="weighted")
        acc_sp = accuracy_score(y_true_sp, y_pred_sp)

        rows.append({"Variant": variant_name, "N": len(df), "Model": name,
                      "Standard F1": f1_std, "Standard Acc": acc_std,
                      "Spatial F1": f1_sp, "Spatial Acc": acc_sp})
    return rows


def main():
    feat = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    df = feat.merge(labels[["Locality_ID", "Draft_Accessibility_Class"]], on="Locality_ID")
    print(f"Base joined dataset: {len(df)} rows")

    full = df.copy()
    quality = df[df["N_Raster_Cells_Imputed"] < 8].copy()
    high_conf = df[df["N_Raster_Cells_Imputed"] == 0].copy()

    print(f"Full:            {len(full)} rows")
    print(f"Quality subset:  {len(quality)} rows (excludes {len(full) - len(quality)} all-NoData rows)")
    print(f"High-confidence: {len(high_conf)} rows (zero raster cells imputed)")

    for name, subset in [("Quality", quality), ("High-confidence", high_conf)]:
        dist = subset["Draft_Accessibility_Class"].value_counts()
        print(f"\n{name} class distribution:\n{dist}")

    all_rows = []
    for name, subset in [("Full (375)", full), ("Quality (362)", quality), ("High-confidence (161)", high_conf)]:
        print(f"\n=== Running benchmark on: {name} ===")
        all_rows += benchmark(subset, name)

    results = pd.DataFrame(all_rows)
    print("\n\n=== ABLATION RESULTS: Spatial Block CV F1 by dataset variant and model ===")
    pivot = results.pivot(index="Model", columns="Variant", values="Spatial F1")
    print(pivot.round(4))

    print("\n=== Best model per variant (by Spatial F1) ===")
    for variant in results["Variant"].unique():
        sub = results[results["Variant"] == variant]
        best = sub.loc[sub["Spatial F1"].idxmax()]
        print(f"  {variant:24s}: {best['Model']:22s} Spatial F1={best['Spatial F1']:.4f} Acc={best['Spatial Acc']:.4f}")

    out_csv = os.path.join(HERE, "..", "data", "model_outputs", "ablation_results.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nSaved full results table to {out_csv}")


if __name__ == "__main__":
    main()
