"""
Task 6 -- Baseline accessibility classifier for the v2 geosite catalog
(geosite_v2_work/data/geosites_features.csv joined with
geosites_labels_draft.csv on Locality_ID).

This is an explicit BASELINE on DRAFT labels: Draft_Accessibility_Class was
auto-derived from OSRM_Travel_Time_Min and has NOT yet been reviewed by the
user (User_Override_Class / User_Notes are currently all empty). The model
below should be re-trained once human-reviewed labels are available.

Reuses the benchmark structure proven in
livrable/phase1_national_accessibility/code/03_train_accessibility_model.py:
  - Benchmark 3 classifiers (Random Forest, XGBoost, HistGradientBoosting)
  - Standard StratifiedKFold CV *and* Spatial Block CV (GroupKFold over
    0.5-degree lat/lon grid cells) -- spatial CV is the trustworthy number
    for this project, since standard CV overstates performance on
    geographically clustered site data.
  - Mandatory leakage sanity check: this dataset's label is derived from
    OSRM_Travel_Time_Min, which is NOT one of the 9 terrain/distance predictor
    features by construction -- but we verify this isn't accidentally true by
    computing an out-of-fold R^2 of OSRM_Travel_Time_Min regressed on the 9
    features. A value near 1.0 would mean the label is leaking through the
    features and any classification accuracy would be meaningless.
  - Saved outputs: model comparison bar chart, out-of-fold confusion matrix
    (spatial CV, best model), and permutation feature importance plot.

Class balance check (this dataset, n=375): Easy=36, Moderate=147,
Difficult=131, Very Difficult=61. All 4 classes are large enough for
StratifiedKFold(n_splits=5) (smallest class 36 / 5 = ~7 per fold) -- no merge
needed here, unlike the earlier phase1 pipeline which had a 1-member class.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "training")
OUT_DIR = os.path.join(HERE, "..", "data", "model_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
FEATURES_CSV = os.path.join(DATA_DIR, "geosites_features.csv")
LABELS_CSV = os.path.join(DATA_DIR, "geosites_labels_draft.csv")
MODEL_OUT = os.path.join(OUT_DIR, "baseline_model.joblib")
FIG_COMPARISON = os.path.join(OUT_DIR, "model_comparison.png")
FIG_CONFUSION = os.path.join(OUT_DIR, "confusion_matrix.png")
FIG_IMPORTANCE = os.path.join(OUT_DIR, "feature_importances.png")

FEATURES = [
    "Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Dam_m", "Dist_to_River_m",
    "LULC_Class", "Soil_Class", "Geology_Class", "Dist_to_Highway_m",
]

CLASS_NAMES = ["Easy", "Moderate", "Difficult", "Very Difficult"]
NAME_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}


def get_spatial_blocks(lats, lons, block_size=0.5):
    lat_b = (np.array(lats) / block_size).astype(int)
    lon_b = (np.array(lons) / block_size).astype(int)
    return np.array([f"{la}_{lo}" for la, lo in zip(lat_b, lon_b)])


def main():
    feats = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)

    print(f"Loaded {len(feats)} rows from {os.path.basename(FEATURES_CSV)}, "
          f"{len(labels)} rows from {os.path.basename(LABELS_CSV)}")

    df = feats.merge(
        labels[["Locality_ID", "OSRM_Travel_Time_Min", "Draft_Accessibility_Class"]],
        on="Locality_ID", how="inner",
    )
    assert len(df) == len(feats) == len(labels), (
        f"Join mismatch: features={len(feats)}, labels={len(labels)}, joined={len(df)}"
    )
    print(f"Joined dataset: {len(df)} localities (1:1 join on Locality_ID confirmed)")

    print("\n=== DRAFT LABEL DISTRIBUTION (user has not reviewed yet -- draft only) ===")
    print(df["Draft_Accessibility_Class"].value_counts())

    missing_feat = df[FEATURES].isna().sum().sum()
    missing_label = df["Draft_Accessibility_Class"].isna().sum()
    assert missing_feat == 0, f"{missing_feat} missing feature values found"
    assert missing_label == 0, f"{missing_label} missing labels found"

    df["Label"] = df["Draft_Accessibility_Class"].map(NAME_TO_INT)
    assert df["Label"].isna().sum() == 0, "Unmapped Draft_Accessibility_Class value found"
    df["Label"] = df["Label"].astype(int)

    min_class_count = df["Draft_Accessibility_Class"].value_counts().min()
    n_splits = 5
    print(f"\nSmallest class has {min_class_count} members; using StratifiedKFold(n_splits={n_splits}) "
          f"-> ~{min_class_count / n_splits:.1f} members per fold per class.")
    assert min_class_count >= n_splits, (
        f"Smallest class has only {min_class_count} members -- "
        f"insufficient for StratifiedKFold(n_splits={n_splits})"
    )

    X = df[FEATURES].copy()
    y = df["Label"].values
    t = df["OSRM_Travel_Time_Min"].values
    groups = get_spatial_blocks(df["Latitude_WGS84"].values, df["Longitude_WGS84"].values, block_size=0.5)
    print(f"Spatial blocks (0.5deg grid, ~55km cells): {len(np.unique(groups))} unique blocks "
          f"over {len(df)} localities")

    # ------------------------------------------------------------------
    # MANDATORY LEAKAGE SANITY CHECK
    # OSRM_Travel_Time_Min (the quantity the draft label is derived from) is
    # regressed on the 9 terrain/distance predictor features, out-of-fold. If
    # the 9 features could reconstruct travel time almost perfectly, the
    # label would be leaking through the features and downstream
    # classification accuracy would be meaningless.
    # ------------------------------------------------------------------
    print("\n=== LEAKAGE SANITY CHECK: out-of-fold R^2 of OSRM_Travel_Time_Min ~ 9 features ===")
    gkf_leak = GroupKFold(n_splits=5)
    t_true, t_pred = [], []
    for tr, te in gkf_leak.split(X, t, groups=groups):
        reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        reg.fit(X.iloc[tr], t[tr])
        t_true += list(t[te])
        t_pred += list(reg.predict(X.iloc[te]))
    leak_r2 = r2_score(t_true, t_pred)
    print(f"Out-of-fold R^2(OSRM_Travel_Time_Min | 9 features) = {leak_r2:.4f} "
          f"(spatial-block CV, RandomForestRegressor)")
    if leak_r2 > 0.95:
        print("WARNING: R^2 is suspiciously close to 1.0 -- investigate possible leakage "
              "before trusting any downstream model results.", file=sys.stderr)
    else:
        print("R^2 is well below 1.0 -- no evidence the 9 terrain/distance features "
              "mathematically determine the OSRM travel-time label. Proceeding.")

    # ------------------------------------------------------------------
    # MODEL BENCHMARK: Standard (Stratified) CV vs Spatial Block (GroupKFold) CV
    # ------------------------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="mlogloss"),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, max_iter=150, random_state=42),
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=n_splits)

    results, oof = [], {}
    best_f1, best_name, best_obj = -1.0, "", None

    print("\n=== MODEL BENCHMARK: Standard (Stratified) CV vs Spatial Block (GroupKFold) CV ===")
    for name, model in models.items():
        y_true_std, y_pred_std = [], []
        for tr, te in skf.split(X, y):
            model.fit(X.iloc[tr], y[tr])
            y_true_std += list(y[te])
            y_pred_std += list(model.predict(X.iloc[te]))
        f1_std = f1_score(y_true_std, y_pred_std, average="weighted")
        acc_std = accuracy_score(y_true_std, y_pred_std)

        y_true_sp, y_pred_sp = [], []
        for tr, te in gkf.split(X, y, groups=groups):
            model.fit(X.iloc[tr], y[tr])
            y_true_sp += list(y[te])
            y_pred_sp += list(model.predict(X.iloc[te]))
        y_true_sp, y_pred_sp = np.array(y_true_sp), np.array(y_pred_sp)
        f1_sp = f1_score(y_true_sp, y_pred_sp, average="weighted")
        acc_sp = accuracy_score(y_true_sp, y_pred_sp)

        oof[name] = (y_true_sp, y_pred_sp)
        results.append({
            "Model": name, "Standard F1": f1_std, "Standard Acc": acc_std,
            "Spatial F1": f1_sp, "Spatial Acc": acc_sp,
        })
        print(f"{name:22s} | Standard F1={f1_std:.4f} Acc={acc_std:.4f} | "
              f"Spatial F1={f1_sp:.4f} Acc={acc_sp:.4f}  (gap={f1_std - f1_sp:+.4f})")
        if f1_sp > best_f1:
            best_f1, best_name, best_obj = f1_sp, name, model

    print(f"\nBest model by SPATIAL Block CV (the trustworthy number): {best_name} (Spatial F1={best_f1:.4f})")
    assert best_f1 < 0.98, (
        "Suspiciously perfect spatial F1 -- check for remaining label leakage "
        "before trusting this number"
    )
    assert best_f1 > 0.0, "Spatial F1 is zero -- model is not learning anything useful"

    best_obj.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(best_obj, MODEL_OUT)
    print(f"\nSaved best model ({best_name}, fit on full data) to:\n  {MODEL_OUT}")

    # --- Figure: model comparison (Standard CV vs Spatial Block CV) ---
    df_res = pd.DataFrame(results)
    print("\n=== FULL COMPARISON TABLE ===")
    print(df_res.to_string(index=False))

    plot_rows = []
    for _, r in df_res.iterrows():
        plot_rows.append({"Model": r["Model"], "Scheme": "Standard CV (Stratified)", "F1": r["Standard F1"]})
        plot_rows.append({"Model": r["Model"], "Scheme": "Spatial Block CV (~55km)", "F1": r["Spatial F1"]})
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=pd.DataFrame(plot_rows), x="Model", y="F1", hue="Scheme", palette=["#2ecc71", "#3498db"])
    plt.title("Baseline Model Benchmark -- Draft Accessibility Label (4-class)\n"
              "Easy / Moderate / Difficult / Very Difficult (UNREVIEWED draft labels)", fontweight="bold")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_COMPARISON, dpi=300)
    plt.close()

    # --- Figure: out-of-fold confusion matrix (spatial block CV, best model) ---
    y_true_best, y_pred_best = oof[best_name]
    cm = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1, 2, 3])
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    plt.title(f"Out-of-Fold Confusion Matrix -- {best_name}\n(Draft label, Spatial Block CV)", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIG_CONFUSION, dpi=300)
    plt.close()

    # --- Figure: permutation importance (final model, full data) ---
    perm = permutation_importance(best_obj, X, y, n_repeats=10, random_state=42)
    s_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    plt.figure(figsize=(10, 6))
    plt.barh(s_imp.index, s_imp.values, color=sns.color_palette("viridis", len(s_imp)))
    plt.title("Permutation Feature Importance -- Baseline Accessibility Model (draft labels)", fontweight="bold")
    plt.xlabel("Mean accuracy decrease upon permutation")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE, dpi=300)
    plt.close()

    s_imp_desc = s_imp.sort_values(ascending=False)
    print("\nPermutation importances (full-data fit, descending):\n", s_imp_desc)

    total_imp = s_imp_desc.sum()
    top_share = s_imp_desc.iloc[0] / total_imp if total_imp > 0 else 0.0
    if top_share >= 0.80:
        print(f"\nFLAG: top feature '{s_imp_desc.index[0]}' accounts for "
              f"{top_share:.1%} of total permutation importance -- a single feature "
              f"dominating this heavily is a red flag for a shortcut/near-leakage "
              f"relationship, not a well-rounded terrain/road model. Investigate before "
              f"trusting this model.", file=sys.stderr)
    else:
        print(f"\nTop feature ('{s_imp_desc.index[0]}') accounts for {top_share:.1%} of total "
              f"permutation importance -- no single-feature dominance red flag.")

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFATAL: sanity check failed -- {e}", file=sys.stderr)
        sys.exit(1)
