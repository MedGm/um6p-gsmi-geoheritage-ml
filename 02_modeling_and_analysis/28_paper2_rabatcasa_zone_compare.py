"""
data_audit/28_paper2_rabatcasa_zone_compare.py  (2026-08-23)

Rabat/Casablanca zone comparison for the national mosaic map, same fair
apples-to-apples logic as data_audit/27 Part 2 (Eddakhla): refit BOTH
Rabat-standalone's winning config AND the merged RabatCasablanca pair's
winning config WITH per-site OOF, then compare accuracy restricted to
Rabat-Salé-Kénitra's own sites only.

Difficult target: Rabat-standalone was degenerate (n_pos<3, skipped in
data_audit/27 Part 1) -- only the merged pair is usable, so Difficult's
decision is "merged" by construction, no comparison needed.

Easy target: both options exist (Rabat-standalone acc=0.8095 on N=21;
merged-pair acc=0.7857 on N=28) -- run the restricted comparison to decide.

Output: results/json/other/phase5_paper2_rabatcasa_zone_comparison.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
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
assert len(merged) == 1662

SENTINEL_DIST_M = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
settle_dummies = pd.get_dummies(merged["nearest_settlement_type"], prefix="Settlement").astype(float)
merged = pd.concat([merged, settle_dummies], axis=1)
INFRA_COLS = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"] + list(settle_dummies.columns)

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

MODEL_KINDS = ["RF", "XGB", "GBM", "LGBM"]

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr): return compute_sample_weight("balanced", y_tr)

def fit_predict_proba(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = combined_weight(y[tr])
    if kind == "XGB":
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict(X[te]), m.predict_proba(X[te])

def logo_fold_proba(best_cfgs, X, y, tr, te):
    probs = [fit_predict_proba(k, c, X, y, tr, te)[1] for k, c in best_cfgs]
    return te, np.mean(probs, axis=0)

def logo_cluster_cv_proba(best_cfgs, X, y, groups, n_jobs=-1):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(logo_fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.zeros((len(y), 2))
    for te, p in results: proba[te] = p
    return proba

def dummies_for(sub):
    dom_counts = sub["Geological_Domain"].value_counts()
    rare = dom_counts[dom_counts < 5].index
    sub = sub.copy()
    sub["Domain_grouped"] = sub["Geological_Domain"].where(~sub["Geological_Domain"].isin(rare), "Other").fillna("Unknown")
    dom_dummies = pd.get_dummies(sub["Domain_grouped"], prefix="Domain").astype(float)
    return pd.concat([sub, dom_dummies], axis=1), dom_dummies

zone_comparison = {}

rabat_best = json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_rabat_standalone_results.json")))
rabat_by_target = {r["target"]: r for r in rabat_best}
merged_best = json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_merged_regions_results.json")))
pair_by_target = {r["target"]: r for r in merged_best if r["group"] == "RabatCasablanca"}

for target in ["Difficult", "Easy"]:
    if target not in rabat_by_target:
        # Rabat-standalone was skipped (degenerate class balance) for this target at
        # the current N -- merged pair is the only usable option, no comparison possible.
        zone_comparison[target] = dict(decision="merged", reason="Rabat-standalone degenerate (skipped at current N), no comparison possible")
        log(f"{target}: decision=merged (Rabat-standalone degenerate at current N, only usable option)")
        continue

    log(f"\n=== {target}: Rabat-standalone vs merged-pair, restricted to Rabat's own sites ===")
    rabat_entry = rabat_by_target[target]
    pair_entry = pair_by_target[target]

    # Rabat-standalone: refit its winning variant on Rabat-only data, WITH per-site OOF
    rabat_winner = rabat_entry["best_variant"]
    sub_rabat = merged[merged["Region"] == "Rabat-Salé-Kénitra"].reset_index(drop=True)
    sub_rabat, dom_dummies_rabat = dummies_for(sub_rabat)
    groups_rabat = cluster_of(sub_rabat)
    yb_rabat = (sub_rabat["Expert_Merged"] == target).astype(int).values
    cols_rabat = {"Baseline": FEATURES_BASE, "Domain": FEATURES_BASE + list(dom_dummies_rabat.columns), "Infra": FEATURES_BASE + INFRA_COLS}[rabat_winner]
    cfgs_rabat = rabat_entry["variants"][rabat_winner]["best_configs"]
    rabat_thr = rabat_entry["variants"][rabat_winner]["tuned_threshold"]
    proba_rabat_standalone = logo_cluster_cv_proba(cfgs_rabat, sub_rabat[cols_rabat].values, yb_rabat, groups_rabat)
    # apply each config's own deployed operating point (default 0.5 vs tuned threshold,
    # whichever the original grid search selected as best -- same definition as best_acc
    # elsewhere), not a bare 0.5 cutoff, so the comparison is a fair like-for-like.
    acc_rabat_default = accuracy_score(yb_rabat, (proba_rabat_standalone[:,1]>=0.5).astype(int))
    acc_rabat_tuned = accuracy_score(yb_rabat, (proba_rabat_standalone[:,1]>=rabat_thr).astype(int))
    acc_rabat_standalone = max(acc_rabat_default, acc_rabat_tuned)
    log(f"  Rabat-standalone ({rabat_winner}) refit acc on its own {len(yb_rabat)} sites: default={acc_rabat_default:.4f} tuned@{rabat_thr}={acc_rabat_tuned:.4f} -> {acc_rabat_standalone:.4f} (script reported {rabat_entry['best_acc']})")

    # Merged pair: refit its winning variant on the pair's data, WITH per-site OOF, then restrict to Rabat's rows
    pair_winner = pair_entry["best_variant"]
    sub_pair = merged[merged["Region"].isin(["Rabat-Salé-Kénitra", "Grand Casablanca-Settat"])].reset_index(drop=True)
    sub_pair, dom_dummies_pair = dummies_for(sub_pair)
    groups_pair = cluster_of(sub_pair)
    yb_pair = (sub_pair["Expert_Merged"] == target).astype(int).values
    cols_pair = {"Baseline": FEATURES_BASE, "Domain": FEATURES_BASE + list(dom_dummies_pair.columns), "Infra": FEATURES_BASE + INFRA_COLS}[pair_winner]
    cfgs_pair = pair_entry["variants"][pair_winner]["best_configs"]
    pair_thr = pair_entry["variants"][pair_winner]["tuned_threshold"]
    proba_pair = logo_cluster_cv_proba(cfgs_pair, sub_pair[cols_pair].values, yb_pair, groups_pair)
    acc_pair_default = accuracy_score(yb_pair, (proba_pair[:,1]>=0.5).astype(int))
    acc_pair_tuned = accuracy_score(yb_pair, (proba_pair[:,1]>=pair_thr).astype(int))
    pair_use_tuned = acc_pair_tuned >= acc_pair_default
    acc_pair_full = max(acc_pair_default, acc_pair_tuned)
    log(f"  Merged-pair ({pair_winner}) refit acc on full {len(yb_pair)}-site pair: default={acc_pair_default:.4f} tuned@{pair_thr}={acc_pair_tuned:.4f} -> {acc_pair_full:.4f} (script reported {pair_entry['best_acc']})")

    # apply the SAME deployed operating point (chosen above on the full pair) to the
    # restricted Rabat-only subset -- the threshold is part of the model, evaluation
    # scope is what's restricted.
    rabat_mask_in_pair = (sub_pair["Region"] == "Rabat-Salé-Kénitra").values
    thr_to_use = pair_thr if pair_use_tuned else 0.5
    pred_pair_on_rabat = (proba_pair[rabat_mask_in_pair, 1] >= thr_to_use).astype(int)
    y_pair_on_rabat = yb_pair[rabat_mask_in_pair]
    acc_pair_on_rabat = accuracy_score(y_pair_on_rabat, pred_pair_on_rabat)
    log(f"  Merged-pair model's accuracy RESTRICTED to Rabat's own {rabat_mask_in_pair.sum()} sites (thr={thr_to_use}): {acc_pair_on_rabat:.4f}")

    winner_for_rabat_zone = "standalone" if acc_rabat_standalone >= acc_pair_on_rabat else "merged"
    log(f"  -> DECISION for {target}: {winner_for_rabat_zone} (standalone={acc_rabat_standalone:.4f} vs merged-on-rabat={acc_pair_on_rabat:.4f})")

    zone_comparison[target] = dict(
        rabat_standalone_acc=round(acc_rabat_standalone,4), rabat_standalone_variant=rabat_winner,
        merged_full_acc=round(acc_pair_full,4), merged_variant=pair_winner,
        merged_acc_on_rabat_only=round(acc_pair_on_rabat,4),
        n_rabat_sites=int(rabat_mask_in_pair.sum()),
        decision=winner_for_rabat_zone,
    )

out_path = os.path.join(BASE, "results/json/other/phase5_paper2_rabatcasa_zone_comparison.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(zone_comparison, f, indent=2)
log(f"\nWrote {out_path}")

log("\nFINAL SUMMARY:")
for target, r in zone_comparison.items():
    print(f"  Rabat/Casa zone {target:10s} decision={r['decision']}")
