"""
code/29_lcp_bmk_slice.py  (2026-08-17)

Follow-up to code/27's LCP feature test, scoped to Béni Mellal-Khénifra
specifically -- per review: the friction-surface hypothesis (karst caves,
river-canyon crossings) was motivated by BMK's documented six-feature blind
spot, not Eddakhla's (flat, low-relief Saharan terrain where a friction-aware
path distance should mathematically collapse to ~straight-line distance,
which likely explains code/27's identical Eddakhla leave-region-out numbers
before/after almost by construction, independent of the feature's value
elsewhere). code/27's leave_region_out() output already gives BMK's number
per variant (no rerun needed, see results/json/training/lcp_feature_test_results.json);
this script adds the complementary LOGO-cluster CV number sliced to BMK's 157
sites specifically -- i.e. with BMK training data included (unlike leave-
region-out), does the feature help predict held-out BMK points better?
Reuses the best_configs already found by code/27's grid search (saved in that
run's json) rather than re-running grid search.

Output: results/json/training/lcp_bmk_slice_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    cols = ["Locality_ID", "Expert_Class"]
    if "Confidence" in labeled.columns: cols.append("Confidence")
    else: labeled["Confidence"] = "Medium"; cols.append("Confidence")
    frames.append(labeled[cols])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
lcp = pd.read_csv(os.path.join(BASE, "data/final/dist_to_road_leastcost_m.csv"))[["Locality_ID", "LeastCost_Effective_m"]]
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").merge(lcp, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged["conf_weight"] = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7)
N = len(merged)
assert N == 733
bmk_mask = (merged["Region"] == "Béni Mellal-Khénifra").values
log(f"N={N}, BMK sites={bmk_mask.sum()}")

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
conf_w = merged["conf_weight"].values
log(f"clusters={len(np.unique(cluster_ids))}")

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr, conf_tr):
    return conf_tr * compute_sample_weight("balanced", y_tr)

def fold_proba(best_cfgs, X, y, cw, tr, te):
    probs = []
    for kind, cfg in best_cfgs:
        m = make_model(kind, cfg)
        sw = combined_weight(y[tr], cw[tr])
        if kind == "XGB":
            pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
            if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te])[:, 1])
    return te, np.mean(probs, axis=0)

def logo_cluster_bmk_acc(cols, y, best_cfgs):
    X = merged[cols].values
    folds = list(LeaveOneGroupOut().split(X, y, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(fold_proba)(best_cfgs, X, y, conf_w, tr, te) for tr, te in folds)
    proba = np.full(N, np.nan)
    for te, p in results: proba[te] = p
    preds = (proba >= 0.5).astype(int)
    bmk_acc = float((preds[bmk_mask] == y[bmk_mask]).mean())
    national_acc = float(np.nanmean(preds == y))
    return bmk_acc, national_acc

lcp_results = json.load(open(os.path.join(BASE, "results/json/training/lcp_feature_test_results.json")))
g0a = json.load(open(os.path.join(BASE, "results/json/training/final_v2_results_N733.json")))

feature_sets = {
    "LCP_replace": ["LeastCost_Effective_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"],
    "LCP_add": FEATURES_BASE + ["LeastCost_Effective_m"],
}
out = {}
for target_name in ["difficult", "easy"]:
    y = (merged["Expert_Merged"] == ("Difficult" if target_name == "difficult" else "Easy")).astype(int).values
    out[target_name] = {}
    for fset_name, cols in feature_sets.items():
        tag = f"{fset_name}_{target_name}"
        best_cfgs = [(k, c) for k, c in lcp_results[tag]["best_configs"]]
        log(f"Refitting {tag} with saved best_configs (BMK-sliced LOGO-cluster CV) ...")
        bmk_acc, national_acc = logo_cluster_bmk_acc(cols, y, best_cfgs)
        out[target_name][fset_name] = {"bmk_acc_logo_cluster": round(bmk_acc, 4), "national_acc_logo_cluster": round(national_acc, 4)}
        log(f"  {tag}: BMK acc={bmk_acc:.4f}  national acc={national_acc:.4f}")

log("Recomputing BMK-sliced baseline (G0a, 6 base features) for direct comparison ...")
for target_name in ["difficult", "easy"]:
    y = (merged["Expert_Merged"] == ("Difficult" if target_name == "difficult" else "Easy")).astype(int).values
    best_cfgs = [(k, c) for k, c in g0a[f"G0a_{target_name}"]["best_configs"]]
    bmk_acc, national_acc = logo_cluster_bmk_acc(FEATURES_BASE, y, best_cfgs)
    out[target_name]["G0a_baseline"] = {"bmk_acc_logo_cluster": round(bmk_acc, 4), "national_acc_logo_cluster": round(national_acc, 4)}
    log(f"  G0a_{target_name}: BMK acc={bmk_acc:.4f}  national acc={national_acc:.4f}")

out_path = os.path.join(BASE, "results", "json", "training", "lcp_bmk_slice_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"Wrote {out_path}")

log("SUMMARY (BMK-sliced LOGO-cluster CV, n=157):")
for target_name in ["difficult", "easy"]:
    base = out[target_name]["G0a_baseline"]["bmk_acc_logo_cluster"]
    print(f"  {target_name}: G0a baseline BMK acc={base}")
    for fset_name in feature_sets:
        acc = out[target_name][fset_name]["bmk_acc_logo_cluster"]
        print(f"    {fset_name}: {acc}  (delta {100*(acc-base):+.2f}pp)")
