"""
data_audit/18_infra_mcnemar.py  (2026-08-22)

McNemar check on InfraAdd_939 vs Baseline_939 -- is the day's biggest delta
(Difficult +1.91pp, Easy +2.66pp) real, or does it join Domain_939 (p=0.40)
and RoutingReplace (p=0.31 national) as another non-significant bump?

Reuses Baseline_939's already-saved per-site OOF (phase5_difficult_oof_per_site.csv,
phase5_easy_oof_per_site.csv) and refits ONLY InfraAdd_939's LOGO-cluster CV
(reusing best_configs already saved in phase5_infra_feature_results.json, no
grid search rerun) to get InfraAdd_939's real per-site predictions.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
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
    on="Locality_ID", how="inner").merge(infra, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

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

infra_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_infra_feature_results.json")))

out = {}
for target_name, target_class in [("difficult", "Difficult"), ("easy", "Easy")]:
    y = (merged["Expert_Merged"] == target_class).astype(int).values
    best_cfgs = infra_results[f"InfraAdd_939_{target_name}"]["best_configs"]
    X = merged[FEATURES_INFRA].values

    log(f"[{target_name}] Refitting InfraAdd_939 LOGO-cluster CV OOF (reused best_configs) ...")
    proba_infra = logo_cluster_oof(X, y, best_cfgs)
    acc_infra = float(np.nanmean((proba_infra >= 0.5).astype(int) == y))
    log(f"  InfraAdd_939_{target_name} acc={acc_infra:.4f} (grid-search run reported {infra_results[f'InfraAdd_939_{target_name}']['acc_logo_cluster']})")

    baseline_oof = pd.read_csv(os.path.join(BASE, f"results/json/other/phase5_{target_name}_oof_per_site.csv"))[["Locality_ID", "proba"]].rename(columns={"proba": "proba_baseline"})
    tmp = merged[["Locality_ID"]].copy()
    tmp["proba_infra"] = proba_infra
    tmp["y"] = y
    tmp = tmp.merge(baseline_oof, on="Locality_ID", how="left")

    correct_base = (tmp["proba_baseline"] >= 0.5).astype(int) == tmp["y"]
    correct_infra = (tmp["proba_infra"] >= 0.5).astype(int) == tmp["y"]
    valid = tmp["proba_baseline"].notna() & tmp["proba_infra"].notna()

    a, b = correct_base[valid].values, correct_infra[valid].values
    table = pd.crosstab(pd.Series(a), pd.Series(b)).reindex(index=[False, True], columns=[False, True], fill_value=0).values
    res = mcnemar(table, exact=False, correction=True)
    n10 = int(((a == True) & (b == False)).sum())
    n01 = int(((a == False) & (b == True)).sum())
    log(f"  McNemar Baseline_939 vs InfraAdd_939 ({target_name}): chi2={res.statistic:.4f} p={res.pvalue:.4f} "
        f"(baseline_right_infra_wrong={n10}, baseline_wrong_infra_right={n01}, n={valid.sum()})")
    out[target_name] = {"acc_baseline": round(float(a.mean()),4), "acc_infra": round(float(b.mean()),4),
                         "chi2": round(float(res.statistic),4), "p": round(float(res.pvalue),4),
                         "n10": n10, "n01": n01}

out_path = os.path.join(BASE, "results/json/other/phase5_infra_mcnemar_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"Wrote {out_path}")
