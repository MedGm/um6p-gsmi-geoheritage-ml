"""
data_audit/29_paper2_map_oof.py  (2026-08-23)

Fixes a real bug flagged by user review: the per-region map's "misclassified"
ring markers (report/scripts/make_paper2_region_maps.py) were derived by
looking up the NEAREST GRID CELL's class in a coarse (130x130 over the whole
region bbox) in-sample raster -- resolution far coarser than the 500m
LOGO-cluster radius used everywhere else in this project, so multiple sites
a few km apart (different true classes) routinely collapsed onto the same
grid cell, which can only show one class. Combined with the map raster
sometimes being a Baseline-substituted refit standing in for the TRUE
winning Domain/Infra variant, the ring count had no necessary relationship
to the actually-reported LOGO-CV accuracy -- explaining why maps for
genuinely good models (e.g. rabatcasa Easy, acc=0.786) looked like almost
everything was wrong.

This script computes the actual per-site LOGO-cluster CV out-of-fold
prediction for every unit/target, using the EXACT winning feature-set
composition and best_configs already selected in data_audit/24, 25, 27 --
i.e. it reproduces the already-reported accuracy numbers exactly, and gives
a resolution-independent, feature-set-consistent ground truth for which
sites are actually misclassified. The render script now rings sites using
this file instead of a raster lookup.

Output: results/json/other/phase5_paper2_map_oof.json
  { unit_key: { target: { locality_ids: [...], true_int: [...], pred_int: [...], acc: float } } }
"""
import glob, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
assert len(merged) == 939

SENTINEL_DIST_M = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
settle_dummies_full = pd.get_dummies(merged["nearest_settlement_type"], prefix="Settlement").astype(float)
merged = pd.concat([merged, settle_dummies_full], axis=1)
INFRA_COLS_FULL = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"] + list(settle_dummies_full.columns)

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

def fit_predict_proba(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = compute_sample_weight("balanced", y[tr])
    if kind == "XGB":
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict_proba(X[te])

def logo_fold_proba(best_cfgs, X, y, tr, te):
    probs = [fit_predict_proba(k, c, X, y, tr, te) for k, c in best_cfgs]
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

best_feature = {(r["region"], r["target"]): r for r in json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_best_feature_results.json")))}
merged_groups_res = {(r["group"], r["target"]): r for r in json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_merged_regions_results.json")))}

UNITS = {
    "fesmeknes":  dict(regions=["Fés-Meknés"], source="individual"),
    "bmk":        dict(regions=["Béni Mellal-Khénifra"], source="individual"),
    "ttah":       dict(regions=["Tanger-Tétouan-Al Hoceima"], source="individual"),
    "draa":       dict(regions=["Drâa-Tafilalet"], source="individual"),
    "soussmassa": dict(regions=["Souss-Massa"], source="individual"),
    "marrakech":  dict(regions=["Marrakech-Safi"], source="individual"),
    "eddakhla":   dict(regions=["Eddakhla-Oued Eddahab"], source="individual"),
    "south_duo":  dict(regions=["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra"], source="merged", group_key="South_GuelmimLaayoune"),
    "south_trio": dict(regions=["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Eddakhla-Oued Eddahab"], source="merged", group_key="South_GuelmimLaayouneEddakhla"),
    "rabatcasa":  dict(regions=["Rabat-Salé-Kénitra", "Grand Casablanca-Settat"], source="merged", group_key="RabatCasablanca"),
}

CLASS_TO_INT = {"Easy": 0, "Moderate": 1, "Difficult": 2}

out = {}
for key, meta in UNITS.items():
    log(f"=== {key} ===")
    sub = merged[merged["Region"].isin(meta["regions"])].reset_index(drop=True)
    sub, dom_dummies = dummies_for(sub)
    groups = cluster_of(sub)
    out[key] = {}
    proba_by_target = {}
    for target in ["Difficult", "Easy"]:
        entry = best_feature[(meta["regions"][0], target)] if meta["source"] == "individual" else merged_groups_res[(meta["group_key"], target)]
        winner = entry["best_variant"]
        cfgs = entry["variants"][winner]["best_configs"]
        cols = {"Baseline": FEATURES_BASE,
                "Domain": FEATURES_BASE + list(dom_dummies.columns),
                "Infra": FEATURES_BASE + INFRA_COLS_FULL}[winner]
        yb = (sub["Expert_Merged"] == target).astype(int).values
        proba = logo_cluster_cv_proba(cfgs, sub[cols].values, yb, groups)
        # Table 1's best_acc = max(acc_default, acc_tuned) -- use whichever
        # operating point the original grid search actually selected as best,
        # not a bare 0.5 cutoff, so this OOF ground truth matches the reported
        # accuracy (and hence the map ring count) exactly, not approximately.
        variant_res = entry["variants"][winner]
        use_tuned = variant_res["acc_tuned"] > variant_res["acc_default"]
        thr = variant_res["tuned_threshold"] if use_tuned else 0.5
        pred = (proba[:, 1] >= thr).astype(int)
        acc = accuracy_score(yb, pred)
        log(f"  {target}: winner={winner} thr={thr} refit acc={acc:.4f} (reported {entry['best_acc']})")
        out[key][target] = dict(
            locality_ids=sub["Locality_ID"].tolist(),
            true_int=yb.tolist(), pred_int=pred.tolist(),
            winner=winner, acc=round(float(acc), 4), reported_acc=entry["best_acc"],
        )
        proba_by_target[target] = (proba[:, 1], thr)

    # Combine the two OOF binary probabilities into the same 3-class rule used
    # for the map raster (Difficult first, then Easy, else Moderate), so the
    # ring markers are directly comparable to what the map actually shows.
    # Each target uses its own selected operating point (see above), not a
    # blanket 0.5.
    (p_diff, thr_diff), (p_easy, thr_easy) = proba_by_target["Difficult"], proba_by_target["Easy"]
    pred_3class = np.where(p_diff >= thr_diff, 2, np.where(p_easy >= thr_easy, 0, 1))
    true_3class = sub["Expert_Merged"].map(CLASS_TO_INT).values
    n_correct = int((pred_3class == true_3class).sum())
    log(f"  combined 3-class: {n_correct}/{len(sub)} correct ({n_correct/len(sub)*100:.1f}%)")
    out[key]["combined"] = dict(
        locality_ids=sub["Locality_ID"].tolist(),
        true_int=true_3class.tolist(), pred_int=pred_3class.tolist(),
    )

out_path = os.path.join(BASE, "results/json/other/phase5_paper2_map_oof.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"Wrote {out_path}")
