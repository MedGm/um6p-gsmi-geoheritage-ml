"""
02_modeling_and_analysis/35_treevknn_reversal_breakdown.py  (2026-09-02)

Investigates WHY kNN(k=10) and Gaussian Process now beat the tree ensemble
on Difficult, and lose to it on Easy, at N=1662 -- reversing this project's
N=939-era conclusion that no alternative displaces the tree ensemble.
Splits each model's already-computed per-site OOF predictions (script 34
for kNN/GP, scripts 03/07/08's saved OOF for the tree ensemble) by labeling
origin (original_733+el_ouali_2026 = "OLD", batch3_2026 = "NEW") to see
where the reversal actually comes from.

Finding: on Difficult, the tree ensemble over-predicts the positive class
on batch3 (138/722 predicted Difficult vs 100/722 truly Difficult -- the
same global-threshold-miscalibration mechanism already documented in Paper
2's robustness section for Fes-Meknes) while kNN/GP, being local/distance-
based rather than one global decision threshold, track batch3's much lower
true prevalence far more closely (77 and 78 predicted positive respectively)
and land close to batch3's high majority baseline instead of well below it.
On Easy, the tree ensemble captures batch3's abundant, more sharply-
separable Easy sites better than kNN/GP do (its own biggest gain of the
three models, +19.5pp over batch3's local majority), widening an edge over
kNN/GP that already existed on the old data alone.

Output: printed only (see the accompanying chat/report writeup for the
full explanation); source data is already-saved per-site OOF CSVs.
"""
import pandas as pd, numpy as np
from sklearn.metrics import precision_recall_fscore_support

import os
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
knn_gp = pd.read_csv(f"{BASE}/results/json/other/phase5_knn_gp_per_site.csv")
tree_d = pd.read_csv(f"{BASE}/results/json/other/phase5_difficult_oof_per_site.csv")[["Locality_ID","pred","y"]].rename(columns={"pred":"tree_pred_difficult","y":"y_difficult_check"})
tree_e = pd.read_csv(f"{BASE}/results/json/other/phase5_easy_oof_per_site.csv")[["Locality_ID","pred","y"]].rename(columns={"pred":"tree_pred_easy","y":"y_easy_check"})

df = knn_gp.merge(tree_d, on="Locality_ID", how="left").merge(tree_e, on="Locality_ID", how="left")
assert (df["y_difficult"] == df["y_difficult_check"]).all()
assert (df["y_easy"] == df["y_easy_check"]).all()

old_mask = df["origin"].isin(["original_733", "el_ouali_2026"])
new_mask = df["origin"] == "batch3_2026"

for target in ["difficult", "easy"]:
    y = df[f"y_{target}"].values
    print(f"\n=== {target.upper()} ===")
    for label, mask in [("ALL N=1662", np.ones(len(df), dtype=bool)),
                         ("OLD (original+el_ouali, N=940)", old_mask.values),
                         ("NEW (batch3, N=722)", new_mask.values)]:
        sub_y = y[mask]
        tree_acc = (df[f"tree_pred_{target}"].values[mask] == sub_y).mean()
        knn_acc = (df[f"knn_pred_{target}"].values[mask] == sub_y).mean()
        gp_col = df[f"gp_pred_{target}"].values[mask]
        gp_covered = gp_col >= 0
        gp_acc = (gp_col[gp_covered] == sub_y[gp_covered]).mean()
        maj = max(sub_y.mean(), 1 - sub_y.mean())
        print(f"  {label:35s} N={mask.sum():4d}  majority={maj:.4f}  tree={tree_acc:.4f}  kNN={knn_acc:.4f}  GP={gp_acc:.4f} (cov={gp_covered.sum()})")

# McNemar-style discordant counts: tree right / kNN wrong vs tree wrong / kNN right, split by origin
print("\n=== Discordant pairs: tree vs kNN, by origin ===")
for target in ["difficult", "easy"]:
    y = df[f"y_{target}"].values
    tree_correct = df[f"tree_pred_{target}"].values == y
    knn_correct = df[f"knn_pred_{target}"].values == y
    for label, mask in [("OLD", old_mask.values), ("NEW(batch3)", new_mask.values)]:
        n10 = int((tree_correct[mask] & ~knn_correct[mask]).sum())  # tree right, knn wrong
        n01 = int((~tree_correct[mask] & knn_correct[mask]).sum())  # tree wrong, knn right
        print(f"  {target} {label}: tree-right/kNN-wrong={n10}  tree-wrong/kNN-right={n01}  (N={mask.sum()})")

# The actual over-prediction mechanism on batch3's Difficult target: does the
# tree ensemble call "Difficult" too often relative to batch3's true (much
# lower) local rate, while kNN/GP track it more closely?
print("\n=== Difficult target on batch3: predicted-positive rate vs true rate (over-prediction check) ===")
y = df["y_difficult"].values[new_mask.values]
n_true_pos = int(y.sum())
for name, col in [("tree", "tree_pred_difficult"), ("kNN", "knn_pred_difficult"), ("GP", "gp_pred_difficult")]:
    pred = df[col].values[new_mask.values]
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    print(f"  {name}: n_predicted_positive={int((pred==1).sum()):3d}  (true positives={n_true_pos}, N={new_mask.sum()})  "
          f"precision={p[1]:.3f}  recall={r[1]:.3f}")
