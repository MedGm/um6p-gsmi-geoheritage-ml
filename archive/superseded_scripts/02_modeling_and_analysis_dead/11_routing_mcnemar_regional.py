"""
data_audit/11_routing_mcnemar_regional.py  (2026-08-22)

Significance check on the one open thread from the routing-distance test
(data_audit/10): RoutingReplace_939_difficult looked like a national wash
(-1.28pp) but showed a large regional gap-closing in Tanger-Tétouan-Al
Hoceima (-14.8pp -> -7.8pp) and Souss-Massa (-17.9pp -> -6.0pp), masked
nationally by a 12-site noisy region (Laayoune-Sakia El Hamra) cratering.
Question: is that regional pattern real, or does it evaporate under McNemar
like several other promising deltas did today (Domain_939, threshold
tuning)?

Reuses Baseline_939_difficult's already-saved per-site OOF
(results/json/other/phase5_difficult_oof_per_site.csv) and refits ONLY
RoutingReplace_939_difficult's LOGO-cluster CV (reusing its already-found
best_configs from phase5_routing_feature_results.json, no grid search
rerun) to get paired per-site predictions on the same 939 sites.

McNemar run three ways: national (all 939), TTAH+Souss-Massa only (the
regions with the apparent effect), and everywhere else (as a check that the
effect isn't just showing up everywhere restricted to a smaller N).
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
ROUTING_FEATURES = ["Dist_to_Highway_Routing_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
routing = pd.read_csv(os.path.join(BASE, "data/final/dist_to_highway_routing_m.csv"))

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
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + ROUTING_FEATURES[:0] + ["Slope_deg","Ruggedness","Elevation_m","LULC_Friction","Dist_to_Settlement_m"]],
    on="Locality_ID", how="inner").merge(routing, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 939
assert merged["Dist_to_Highway_Routing_m"].notna().all()

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
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def fold_proba(best_cfgs, X, y, tr, te):
    probs = []
    for kind, cfg in best_cfgs:
        m = make_model(kind, cfg)
        sw = combined_weight(y[tr])
        if kind == "XGB":
            pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
            if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te])[:, 1])
    return te, np.mean(probs, axis=0)

def logo_cluster_oof(X, y, best_cfgs):
    folds = list(LeaveOneGroupOut().split(X, y, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.full(len(y), np.nan)
    for te, p in results: proba[te] = p
    return proba

routing_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_routing_feature_results.json")))
best_cfgs = routing_results["RoutingReplace_939_difficult"]["best_configs"]
y = (merged["Expert_Merged"] == "Difficult").astype(int).values
X = merged[ROUTING_FEATURES].values

log("Refitting RoutingReplace_939_difficult LOGO-cluster CV OOF (reused best_configs) ...")
proba_routing = logo_cluster_oof(X, y, best_cfgs)
acc_routing = float(np.nanmean((proba_routing >= 0.5).astype(int) == y))
log(f"  RoutingReplace_939_difficult acc={acc_routing:.4f} (grid-search run reported {routing_results['RoutingReplace_939_difficult']['acc_logo_cluster']})")

baseline_oof = pd.read_csv(os.path.join(BASE, "results/json/other/phase5_difficult_oof_per_site.csv"))[["Locality_ID", "proba", "y"]].rename(columns={"proba": "proba_baseline"})
merged["proba_routing"] = proba_routing
merged = merged.merge(baseline_oof, on="Locality_ID", how="left")
assert (merged["y"] == y).all(), "label mismatch between baseline OOF and this run"

merged["correct_baseline"] = (merged["proba_baseline"] >= 0.5).astype(int) == merged["y"]
merged["correct_routing"] = (merged["proba_routing"] >= 0.5).astype(int) == merged["y"]

def run_mcnemar(sub, label):
    a, b = sub["correct_baseline"].values, sub["correct_routing"].values
    table = pd.crosstab(pd.Series(a), pd.Series(b)).reindex(index=[False, True], columns=[False, True], fill_value=0).values
    res = mcnemar(table, exact=False, correction=True)
    n10 = int(((a == True) & (b == False)).sum())
    n01 = int(((a == False) & (b == True)).sum())
    acc_a, acc_b = a.mean(), b.mean()
    log(f"  {label:30s} N={len(sub):4d} acc_baseline={acc_a:.3f} acc_routing={acc_b:.3f} "
        f"chi2={res.statistic:.4f} p={res.pvalue:.4f} (baseline_right_routing_wrong={n10}, baseline_wrong_routing_right={n01})")

log("\n=== McNemar: Baseline_939_difficult vs RoutingReplace_939_difficult ===")
run_mcnemar(merged, "National (all 939)")
target_regions = merged["Region"].isin(["Tanger-Tétouan-Al Hoceima", "Souss-Massa"])
run_mcnemar(merged[target_regions], "TTAH + Souss-Massa only")
run_mcnemar(merged[~target_regions], "Everywhere else")
