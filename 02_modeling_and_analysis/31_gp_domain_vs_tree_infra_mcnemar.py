"""
02_modeling_and_analysis/31_gp_domain_vs_tree_infra_mcnemar.py  (2026-09-01)

Final redeployment check for the Difficult target. 30 found GP+Domain
(0.8087) is the best Difficult result anywhere in the pipeline, ahead of
the best tree-ensemble option, Tree+Infra (0.7864). Before redeploying,
confirm that gap is real, not noise: refits both configs with per-site
predictions and runs a paired McNemar test on the full N=1662 set.

Caveat carried over from 21/23: GP uses StratifiedGroupKFold(10), not full
LOGO-cluster CV like the tree ensemble, so this isn't a perfectly matched
paired test (different fold structure) -- same disclosed deviation as
every other tree-vs-GP comparison in this pipeline.

Output: results/json/other/gp_domain_vs_tree_infra_mcnemar_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
master = pd.read_excel(os.path.join(BASE, "geosites_master_1667_with_accessibility.xlsx"))
domain_lookup = master[["Locality_ID", "Geological_Domain"]]
infra = pd.read_csv(os.path.join(BASE, "data/final/infra_features.csv"))

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
    on="Locality_ID", how="inner").merge(domain_lookup, on="Locality_ID", how="left").merge(infra, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

dom_counts = merged["Geological_Domain"].value_counts()
rare_domains = dom_counts[dom_counts < 5].index
merged["Geological_Domain_grouped"] = merged["Geological_Domain"].where(
    ~merged["Geological_Domain"].isin(rare_domains), "Other").fillna("Unknown")
domain_dummies = pd.get_dummies(merged["Geological_Domain_grouped"], prefix="Domain").astype(float)
merged = pd.concat([merged, domain_dummies], axis=1)
FEATURES_DOMAIN = FEATURES_BASE + list(domain_dummies.columns)

SENTINEL_DIST_M = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
settle_dummies = pd.get_dummies(merged["nearest_settlement_type"], prefix="Settlement").astype(float)
merged = pd.concat([merged, settle_dummies], axis=1)
INFRA_NUMERIC = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"]
FEATURES_INFRA = FEATURES_BASE + INFRA_NUMERIC + list(settle_dummies.columns)

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
y = (merged["Expert_Merged"] == "Difficult").astype(int).values
log(f"N={N}, clusters={len(np.unique(cluster_ids))}")

# --- GP + Domain, StratifiedGroupKFold(10) (matches 30's protocol) ---
X_dom = StandardScaler().fit_transform(merged[FEATURES_DOMAIN].values)
preds_gp = np.full(N, -1, dtype=int)
kernel = 1.0 * RBF(length_scale=1.0)
gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
t_gp = time.time()
n_fold = 0
for tr, te in gkf.split(X_dom, y, groups=cluster_ids):
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
    gp.fit(X_dom[tr], y[tr])
    preds_gp[te] = gp.predict(X_dom[te])
    n_fold += 1
    log(f"  GP+Domain fold {n_fold}/10 done [{time.time()-t_gp:.0f}s]")
acc_gp = accuracy_score(y, preds_gp)
log(f"GP+Domain: acc={acc_gp:.4f} (30 reported 0.8087)")

# --- Tree ensemble + Infra, full LOGO-cluster CV, reusing 17's best_configs ---
infra_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_infra_feature_results.json")))
cfgs_infra = infra_results["InfraAdd_939_difficult"]["best_configs"]
X_infra = merged[FEATURES_INFRA].values

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def fold_pred(cfgs, X, y, tr, te):
    sw = compute_sample_weight("balanced", y[tr])
    probs = []
    for kind, cfg in cfgs:
        m = make_model(kind, cfg)
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te])[:, 1])
    return te, (np.mean(probs, axis=0) >= 0.5).astype(int)

folds = list(LeaveOneGroupOut().split(X_infra, y, groups=cluster_ids))
results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_pred)(cfgs_infra, X_infra, y, tr, te) for tr, te in folds)
preds_tree_infra = np.zeros(N, dtype=int)
for te, p in results: preds_tree_infra[te] = p
acc_tree_infra = accuracy_score(y, preds_tree_infra)
log(f"Tree+Infra: acc={acc_tree_infra:.4f} (17 reported 0.7864)")

correct_gp = (preds_gp == y)
correct_tree = (preds_tree_infra == y)
table = pd.crosstab(pd.Series(correct_tree), pd.Series(correct_gp)).reindex(
    index=[False, True], columns=[False, True], fill_value=0).values
res = mcnemar(table, exact=False, correction=True)
n10 = int(((correct_tree == True) & (correct_gp == False)).sum())
n01 = int(((correct_tree == False) & (correct_gp == True)).sum())
result = dict(gp_domain_acc=round(acc_gp,4), tree_infra_acc=round(acc_tree_infra,4),
              chi2=round(float(res.statistic),4), p=round(float(res.pvalue),4),
              tree_right_gp_wrong=n10, tree_wrong_gp_right=n01, n=N)
log(f"\nMcNemar GP+Domain vs Tree+Infra: {result}")

out_path = os.path.join(BASE, "results/json/other/gp_domain_vs_tree_infra_mcnemar_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
log(f"Wrote {out_path}")
