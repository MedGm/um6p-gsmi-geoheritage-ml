"""
data_audit/06_batch_origin_check.py  (2026-08-21)

Diagnostic: does the model's per-site accuracy differ between the original
733-label batch and the new 206-label el_ouali_2026 batch? Motivated by a
sharp question -- if more real labels should help the model learn the
pattern, why did Difficult-class performance stay flat/dip while N grew?

Two competing explanations:
  (a) batch-specific label-definition drift -- el_ouali_2026 sites are
      systematically harder for the SAME model to get right than the
      original batch (would show as origin-accuracy gap)
  (b) feature deficit -- Difficult's true cause isn't in the 6 features at
      all, so accuracy is flat-poor across BOTH batches equally (no origin
      gap), and Easy (whose cause IS mostly captured) improves with more
      data regardless of origin.

Reuses best_configs already found by Phase 5's grid search (in
phase5_modeling_results.json) -- refits only the LOGO-cluster CV step to get
real per-site OOF predictions (never saved before), no grid search rerun.
Runs Baseline_939 for both targets (Domain_939 not needed for this check).
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
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

log("Loading N=939 labeled catalog, tagging origin batch ...")
catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))

el_ouali_ids = set(pd.read_csv(
    os.path.join(BASE, "data/final/regional_label_sources/el_ouali_2026_expert_labels.csv"))["Locality_ID"])
batch3_ids = set(pd.read_csv(
    os.path.join(BASE, "data/final/regional_label_sources/batch3_2026-08-28_expert_labels.csv"))["Locality_ID"])

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
merged["origin"] = np.select(
    [merged["Locality_ID"].isin(el_ouali_ids), merged["Locality_ID"].isin(batch3_ids)],
    ["el_ouali_2026", "batch3_2026"],
    default="original_733")
N = len(merged)
assert N == 1662
log(f"N={N} | original_733={ (merged.origin=='original_733').sum() } | el_ouali_2026={ (merged.origin=='el_ouali_2026').sum() } | batch3_2026={ (merged.origin=='batch3_2026').sum() }")

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

modeling_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_modeling_results.json")))
X = merged[FEATURES].values

out = {}
for target_name, target_class in [("difficult", "Difficult"), ("easy", "Easy")]:
    y = (merged["Expert_Merged"] == target_class).astype(int).values
    best_cfgs = modeling_results[f"Baseline_939_{target_name}"]["best_configs"]
    log(f"[{target_name}] Refitting Baseline_939 LOGO-cluster CV OOF (reused best_configs) ...")
    proba = logo_cluster_oof(X, y, best_cfgs)
    pred = (proba >= 0.5).astype(int)
    correct = (pred == y)
    valid = ~np.isnan(proba)

    overall_acc = float(correct[valid].mean())
    log(f"  overall acc={overall_acc:.4f} (grid-search run reported {modeling_results[f'Baseline_939_{target_name}']['acc_logo_cluster']})")

    rows = []
    for origin in ["original_733", "el_ouali_2026", "batch3_2026"]:
        mask = valid & (merged["origin"].values == origin)
        n = int(mask.sum())
        n_pos = int(y[mask].sum())
        acc = float(correct[mask].mean()) if n > 0 else float("nan")
        local_maj = float(max((y[mask]==1).mean(), (y[mask]==0).mean())) if n > 0 else float("nan")
        rows.append(dict(origin=origin, n=n, n_pos=n_pos, acc=round(acc,4),
                          local_majority=round(local_maj,4), gap_pp=round((acc-local_maj)*100,1)))
        log(f"    {origin:16s} N={n:4d} pos={n_pos:4d} acc={acc:.4f} local_maj={local_maj:.4f} gap={100*(acc-local_maj):+.1f}pp")

    out[target_name] = {"overall_acc": round(overall_acc,4), "by_origin": rows}

out_path = os.path.join(BASE, "results", "json", "other", "batch_origin_check_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"Wrote {out_path}")
