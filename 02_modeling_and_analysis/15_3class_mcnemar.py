"""
data_audit/15_3class_mcnemar.py  (2026-08-22)

McNemar check on whether direct 3-class modeling (acc=0.5623) really beats
the two-binary ordinal combination (acc=0.5442) by a statistically real
margin, or joins today's pile of ~1-2pp deltas that don't survive scrutiny
(Domain_939, RoutingReplace both didn't).

Refits ONLY the 3-class LOGO-cluster CV step (reusing best_configs already
saved in phase5_3class_results.json, no grid search rerun) to get real
per-site predictions -- never saved the first time. Recombines the already-
saved Baseline_939 Easy/Difficult per-site OOF via the same Frank & Hall
ordinal formula used earlier this session (no retraining needed for that
side, both binaries' OOF already on disk).
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CLASSES = ["Easy", "Moderate", "Difficult"]

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662
y = merged["Expert_Merged"].map({c: i for i, c in enumerate(CLASSES)}).values

def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cluster_of(sub):
    lat, lon = sub["Latitude_WGS84"].values, sub["Longitude_WGS84"].values
    n = len(sub)
    D = haversine_matrix(lat, lon)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] <= 500:
                rx, ry = find(i), find(j)
                if rx != ry: parent[rx] = ry
    return np.array([find(i) for i in range(n)])

cluster_ids = cluster_of(merged)
log(f"clusters={len(np.unique(cluster_ids))}")

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="mlogloss", num_class=3, **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, objective="multiclass", num_class=3, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def fold_proba(best_cfgs, X, y, tr, te):
    probs = []
    for kind, cfg in best_cfgs:
        m = make_model(kind, cfg)
        sw = combined_weight(y[tr])
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te]))
    return te, np.mean(probs, axis=0)

def logo_cluster_oof(X, y, best_cfgs):
    folds = list(LeaveOneGroupOut().split(X, y, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.zeros((len(y), 3))
    for te, p in results: proba[te] = p
    return proba

results_3class = json.load(open(os.path.join(BASE, "results/json/training/phase5_3class_results.json")))
best_cfgs = results_3class["best_configs"]
X = merged[FEATURES_BASE].values

log("Refitting 3-class LOGO-cluster CV OOF (reused best_configs, saving per-site) ...")
proba_3class = logo_cluster_oof(X, y, best_cfgs)
pred_3class = proba_3class.argmax(axis=1)
acc_3class = accuracy_score(y, pred_3class)
log(f"  3-class acc={acc_3class:.4f} (grid-search run reported {results_3class['acc_logo_cluster']})")

merged["pred_3class"] = pred_3class
merged["y_idx"] = y
merged["correct_3class"] = merged["pred_3class"] == merged["y_idx"]

# --- Recombine the already-saved binary OOF via Frank & Hall ordinal formula ---
diff = pd.read_csv(os.path.join(BASE, "results/json/other/phase5_difficult_oof_per_site.csv"))[["Locality_ID", "proba"]].rename(columns={"proba": "p_difficult"})
easy = pd.read_csv(os.path.join(BASE, "results/json/other/phase5_easy_oof_per_site.csv"))[["Locality_ID", "proba"]].rename(columns={"proba": "p_easy"})
merged = merged.merge(diff, on="Locality_ID", how="left").merge(easy, on="Locality_ID", how="left")

p_easy = merged["p_easy"].values
p_diff = merged["p_difficult"].values
p_mod = np.clip(1 - p_easy - p_diff, 0, None)
totals = p_easy + p_mod + p_diff
probs_ordinal = np.column_stack([p_easy/totals, p_mod/totals, p_diff/totals])
pred_ordinal = probs_ordinal.argmax(axis=1)
merged["pred_ordinal"] = pred_ordinal
merged["correct_ordinal"] = merged["pred_ordinal"] == merged["y_idx"]
acc_ordinal = merged["correct_ordinal"].mean()
log(f"  ordinal (two-binary Frank&Hall) acc={acc_ordinal:.4f} (session's earlier reported value: 0.5442)")

a = merged["correct_3class"].values
b = merged["correct_ordinal"].values
table = pd.crosstab(pd.Series(a), pd.Series(b)).reindex(index=[False, True], columns=[False, True], fill_value=0).values
res = mcnemar(table, exact=False, correction=True)
n10 = int(((a == True) & (b == False)).sum())
n01 = int(((a == False) & (b == True)).sum())
log(f"\nMcNemar (3-class direct vs two-binary ordinal): chi2={res.statistic:.4f} p={res.pvalue:.4f} "
    f"(3class_right_ordinal_wrong={n10}, 3class_wrong_ordinal_right={n01}, n={len(merged)})")
