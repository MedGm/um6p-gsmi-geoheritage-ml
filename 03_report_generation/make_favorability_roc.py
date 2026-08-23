"""
Genuine ROC curve for the geosite-location favorability model (v4, AUC 0.956),
recomputed from real out-of-fold predictions under the same 0.5x0.5deg spatial-
block GroupKFold CV as code/34_retrain_favorability_v4.py (identical RF params:
max_depth=14, min_samples_leaf=8, n_estimators=300, random_state=42) -- code/34
itself did not save its out-of-fold probabilities, only the AUC number, so this
script reruns just the CV step (no network refetch: reuses the already-saved
feature table data/model_outputs/geosite_presence_background_pilot_v4.csv) and
verifies the recomputed AUC matches 0.9558 exactly before plotting, so the curve
shown is provably the same one behind the reported number, not a stand-in.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b",
    "text.color": "#2b2b2b",
    "xtick.color": "#2b2b2b",
    "ytick.color": "#2b2b2b",
})

ACCENT = "#2B5F72"
NEUTRAL = "#A6A6A6"

df = pd.read_csv(os.path.join(FW, "data", "model_outputs", "geosite_presence_background_pilot_v4.csv"))
FEATURES = ["Geology_Class", "Soil_Class", "Elevation_m", "Slope_deg", "Ruggedness",
            "Dist_to_Settlement_m", "LULC_Friction", "Geology_Class_Missing", "Soil_Class_Missing"]
X = df[FEATURES].values
y = df["presence"].values

df["block_lat"] = np.floor(df["Latitude_WGS84"] / 0.5).astype(int)
df["block_lon"] = np.floor(df["Longitude_WGS84"] / 0.5).astype(int)
groups = (df["block_lat"].astype(str) + "_" + df["block_lon"].astype(str)).values

gkf = GroupKFold(n_splits=5)
probs = np.zeros(len(y))
for tr, te in gkf.split(X, y, groups=groups):
    m = RandomForestClassifier(max_depth=14, min_samples_leaf=8, n_estimators=300, n_jobs=-1, random_state=42)
    m.fit(X[tr], y[tr])
    probs[te] = m.predict_proba(X[te])[:, 1]

auc = roc_auc_score(y, probs)
assert abs(auc - 0.9558) < 0.001, f"AUC mismatch: got {auc}, expected ~0.9558 -- do not plot an unverified curve"
print(f"AUC recomputed: {auc:.4f} (matches reported 0.956)", flush=True)

fpr, tpr, _ = roc_curve(y, probs)
auc_str = f"{auc:.3f}".replace(".", ",")

def plot_roc(auc_label, model_label, chance_label, xlabel, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=1.8, zorder=3, label=f"{model_label} (AUC = {auc_label})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2, label=chance_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name))
    plt.close(fig)
    print(f"Saved {out_name}")

plot_roc(f"{auc:.3f}", "Model", "Chance (AUC = 0.500)", "False positive rate", "True positive rate",
          "favorability_roc_curve.pdf")
plot_roc(auc_str, "Mod\\`ele", "Hasard (AUC = 0,500)", "Taux de faux positifs", "Taux de vrais positifs",
          "favorability_roc_curve_fr.pdf")
