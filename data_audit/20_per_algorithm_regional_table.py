"""
data_audit/20_per_algorithm_regional_table.py  (2026-08-22)

PV section 3 ("Tableau de Synthèse"): per-region comparison table broken out
by INDIVIDUAL algorithm (RF, XGBoost, Gradient Boosting, LightGBM), not the
soft-voted 4-model ensemble reported everywhere else this session. Reuses
best_configs already found and saved in phase5_regional_models_results.json
(data_audit/05) -- refits each algorithm's OWN LOGO-cluster CV separately
per region/target, no grid search rerun.

Output: results/json/training/phase5_per_algorithm_regional_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

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
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")

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

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def fit_predict(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = combined_weight(y[tr])
    if kind == "XGB":
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return te, m.predict(X[te])

def logo_cluster_acc_single_model(kind, cfg, X, y, groups):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fit_predict)(kind, cfg, X, y, tr, te) for tr, te in folds)
    preds = np.zeros(len(y), dtype=int)
    for te, p in results: preds[te] = p
    return accuracy_score(y, preds)

regional_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_regional_models_results.json")))

rows = []
for entry in regional_results:
    if entry.get("skipped"):
        continue
    reg, target = entry["region"], entry["target"]
    best_cfgs = entry["best_configs"]
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    Xr = sub[FEATURES].values
    yb = (sub["Expert_Merged"] == target).astype(int).values
    sub_clusters = cluster_of(sub)
    log(f"{reg} / {target} (N={len(sub)}) ...")
    row = {"region": reg, "target": target, "N": len(sub), "n_pos": int(yb.sum()),
           "local_majority": entry["local_majority"], "ensemble_best_acc": entry["best_acc"]}
    for kind, cfg in best_cfgs:
        acc = logo_cluster_acc_single_model(kind, cfg, Xr, yb, sub_clusters)
        row[kind] = round(acc, 4)
        log(f"  {kind}: {acc:.4f}")
    rows.append(row)

out_path = os.path.join(BASE, "results", "json", "training", "phase5_per_algorithm_regional_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
log(f"Wrote {out_path}")

df = pd.DataFrame(rows)
log("\nFull table:")
print(df.to_string(index=False))
