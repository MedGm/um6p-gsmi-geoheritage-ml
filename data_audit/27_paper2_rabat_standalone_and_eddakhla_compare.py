"""
data_audit/27_paper2_rabat_standalone_and_eddakhla_compare.py  (2026-08-23)

Two pieces needed to decide the national mosaic map's config for the south
and Rabat/Casablanca zones (user's explicit call, 2026-08-23):

1. Rabat-Salé-Kénitra standalone (N=21) -- same 3-variant best-feature-set
   pipeline as data_audit/24, tested alone as the alternative to the merged
   Rabat+Casablanca-Settat pair.
2. South-zone comparison restricted to Eddakhla's own sites: refits BOTH
   Eddakhla-standalone's winning config AND the South-trio's winning config
   WITH per-site OOF saved this time (neither prior run saved it), then
   compares accuracy on Eddakhla's own sites only -- the fair apples-to-
   apples test for which config should represent the south zone on the map.

Output: results/json/training/phase5_paper2_rabat_standalone_results.json
        results/json/other/phase5_paper2_eddakhla_zone_comparison.json
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
assert len(merged) == 939

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

RF_GRID  = [dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl)
            for ne in [100, 200, 400] for md in [3,4,5,6,7,8,None] for msl in [1,2,3,5]]
XGB_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,150,250] for md in [3,4,5,6] for lr in [0.03,0.08,0.15]]
GBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,200] for md in [2,3,4] for lr in [0.05,0.1]]
LGBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr, num_leaves=nl)
             for ne in [100,200] for md in [3,5,-1] for lr in [0.05,0.1] for nl in [15,31]]
MODEL_KINDS = ["RF", "XGB", "GBM", "LGBM"]
GRIDS = {"RF": RF_GRID, "XGB": XGB_GRID, "GBM": GBM_GRID, "LGBM": LGBM_GRID}

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

def one_fit_acc(kind, cfg, X, y, tr, te):
    pred, _ = fit_predict_proba(kind, cfg, X, y, tr, te)
    return accuracy_score(y[te], pred)

def grid_search(X, y, groups, n_repeats=5, n_splits=5, n_jobs=-1):
    tasks = []
    for kind in MODEL_KINDS:
        for cfg in GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_fit_acc)(kind, cfg, X, y, tr, te) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v)} for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def best_cfgs_from_grid(df_grid):
    return [(k, json.loads(df_grid[df_grid.model == k].iloc[0].config)) for k in MODEL_KINDS]

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

def best_threshold(y_true, proba_pos):
    best_t, best_ba = 0.5, balanced_accuracy_score(y_true, (proba_pos>=0.5).astype(int))
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba_pos>=t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba

def run_variant(Xr, yb, groups):
    df_grid = grid_search(Xr, yb, groups)
    best = best_cfgs_from_grid(df_grid)
    proba = logo_cluster_cv_proba(best, Xr, yb, groups)
    acc_default = accuracy_score(yb, (proba[:,1]>=0.5).astype(int))
    t_opt, _ = best_threshold(yb, proba[:,1])
    acc_tuned = accuracy_score(yb, (proba[:,1]>=t_opt).astype(int))
    return dict(acc_default=round(acc_default,4), acc_tuned=round(acc_tuned,4),
                tuned_threshold=round(t_opt,2), best_acc=round(max(acc_default,acc_tuned),4),
                best_configs=best), proba

def dummies_for(sub):
    dom_counts = sub["Geological_Domain"].value_counts()
    rare = dom_counts[dom_counts < 5].index
    sub = sub.copy()
    sub["Domain_grouped"] = sub["Geological_Domain"].where(~sub["Geological_Domain"].isin(rare), "Other").fillna("Unknown")
    dom_dummies = pd.get_dummies(sub["Domain_grouped"], prefix="Domain").astype(float)
    return pd.concat([sub, dom_dummies], axis=1), dom_dummies

# =====================================================================
# PART 1: Rabat-Salé-Kénitra standalone
# =====================================================================
log("=== PART 1: Rabat-Salé-Kénitra standalone (N=21) ===")
sub = merged[merged["Region"] == "Rabat-Salé-Kénitra"].reset_index(drop=True)
sub, dom_dummies = dummies_for(sub)
groups = cluster_of(sub)

rabat_results = []
for target in ["Difficult", "Easy"]:
    yb = (sub["Expert_Merged"] == target).astype(int).values
    n, n_pos = len(yb), int(yb.sum())
    maj = max((yb==1).mean(), (yb==0).mean())
    log(f"Rabat-Salé-Kénitra / {target}: N={n} pos={n_pos} local_maj={maj:.3f}")
    if n_pos < 3 or n_pos > n - 3:
        log("  SKIPPED -- degenerate class balance")
        continue
    variants = {}
    for vname, cols in [("Baseline", FEATURES_BASE), ("Domain", FEATURES_BASE + list(dom_dummies.columns)), ("Infra", FEATURES_BASE + INFRA_COLS)]:
        Xr = sub[cols].values
        log(f"  running {vname} variant ...")
        res, _ = run_variant(Xr, yb, groups)
        variants[vname] = res
    best_variant = max(variants, key=lambda v: variants[v]["best_acc"])
    log(f"  Rabat-Salé-Kénitra/{target}: Baseline={variants['Baseline']['best_acc']} Domain={variants['Domain']['best_acc']} Infra={variants['Infra']['best_acc']} -> BEST={best_variant}")
    rabat_results.append(dict(region="Rabat-Salé-Kénitra", target=target, N=n, n_pos=n_pos, local_majority=round(maj,3),
                               variants=variants, best_variant=best_variant, best_acc=variants[best_variant]["best_acc"]))

out_path1 = os.path.join(BASE, "results/json/training/phase5_paper2_rabat_standalone_results.json")
with open(out_path1, "w", encoding="utf-8") as f:
    json.dump(rabat_results, f, indent=2, default=str)
log(f"Wrote {out_path1}")

# =====================================================================
# PART 2: South zone comparison restricted to Eddakhla's own sites
# =====================================================================
log("\n=== PART 2: South zone comparison (Eddakhla's own sites only) ===")
eddakhla_best = json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_best_feature_results.json")))
eddakhla_best = {(r["region"], r["target"]): r for r in eddakhla_best if r["region"] == "Eddakhla-Oued Eddahab"}
trio_best = json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_merged_regions_results.json")))
trio_best = {r["target"]: r for r in trio_best if r["group"] == "South_GuelmimLaayouneEddakhla"}

zone_comparison = {}
for target in ["Difficult", "Easy"]:
    log(f"\n--- {target} ---")
    # Eddakhla-standalone: refit its winning variant on Eddakhla-only data, WITH per-site OOF
    edd_entry = eddakhla_best[("Eddakhla-Oued Eddahab", target)]
    edd_winner = edd_entry["best_variant"]
    sub_edd = merged[merged["Region"] == "Eddakhla-Oued Eddahab"].reset_index(drop=True)
    sub_edd, dom_dummies_edd = dummies_for(sub_edd)
    groups_edd = cluster_of(sub_edd)
    yb_edd = (sub_edd["Expert_Merged"] == target).astype(int).values
    cols_edd = {"Baseline": FEATURES_BASE, "Domain": FEATURES_BASE + list(dom_dummies_edd.columns), "Infra": FEATURES_BASE + INFRA_COLS}[edd_winner]
    cfgs_edd = edd_entry["variants"][edd_winner]["best_configs"]
    proba_edd_standalone = logo_cluster_cv_proba(cfgs_edd, sub_edd[cols_edd].values, yb_edd, groups_edd)
    acc_edd_standalone = accuracy_score(yb_edd, (proba_edd_standalone[:,1]>=0.5).astype(int))
    log(f"  Eddakhla-standalone ({edd_winner}) refit acc on its own {len(yb_edd)} sites: {acc_edd_standalone:.4f} (script reported {edd_entry['best_acc']})")

    # Trio: refit its winning variant on the trio's data, WITH per-site OOF, then restrict to Eddakhla's rows
    trio_entry = trio_best[target]
    trio_winner = trio_entry["best_variant"]
    sub_trio = merged[merged["Region"].isin(["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Eddakhla-Oued Eddahab"])].reset_index(drop=True)
    sub_trio, dom_dummies_trio = dummies_for(sub_trio)
    groups_trio = cluster_of(sub_trio)
    yb_trio = (sub_trio["Expert_Merged"] == target).astype(int).values
    cols_trio = {"Baseline": FEATURES_BASE, "Domain": FEATURES_BASE + list(dom_dummies_trio.columns), "Infra": FEATURES_BASE + INFRA_COLS}[trio_winner]
    cfgs_trio = trio_entry["variants"][trio_winner]["best_configs"]
    proba_trio = logo_cluster_cv_proba(cfgs_trio, sub_trio[cols_trio].values, yb_trio, groups_trio)
    acc_trio_full = accuracy_score(yb_trio, (proba_trio[:,1]>=0.5).astype(int))
    log(f"  Trio ({trio_winner}) refit acc on full {len(yb_trio)}-site trio: {acc_trio_full:.4f} (script reported {trio_entry['best_acc']})")

    eddakhla_mask_in_trio = (sub_trio["Region"] == "Eddakhla-Oued Eddahab").values
    pred_trio_on_eddakhla = (proba_trio[eddakhla_mask_in_trio, 1] >= 0.5).astype(int)
    y_trio_on_eddakhla = yb_trio[eddakhla_mask_in_trio]
    acc_trio_on_eddakhla = accuracy_score(y_trio_on_eddakhla, pred_trio_on_eddakhla)
    log(f"  Trio model's accuracy RESTRICTED to Eddakhla's own {eddakhla_mask_in_trio.sum()} sites: {acc_trio_on_eddakhla:.4f}")

    winner_for_eddakhla_zone = "standalone" if acc_edd_standalone >= acc_trio_on_eddakhla else "trio"
    log(f"  -> DECISION for {target}: {winner_for_eddakhla_zone} (standalone={acc_edd_standalone:.4f} vs trio-on-eddakhla={acc_trio_on_eddakhla:.4f})")

    zone_comparison[target] = dict(
        eddakhla_standalone_acc=round(acc_edd_standalone,4), eddakhla_standalone_variant=edd_winner,
        trio_full_acc=round(acc_trio_full,4), trio_variant=trio_winner,
        trio_acc_on_eddakhla_only=round(acc_trio_on_eddakhla,4),
        n_eddakhla_sites=int(eddakhla_mask_in_trio.sum()),
        decision=winner_for_eddakhla_zone,
    )

out_path2 = os.path.join(BASE, "results/json/other/phase5_paper2_eddakhla_zone_comparison.json")
with open(out_path2, "w", encoding="utf-8") as f:
    json.dump(zone_comparison, f, indent=2)
log(f"\nWrote {out_path2}")

log("\nFINAL SUMMARY:")
for r in rabat_results:
    print(f"  Rabat-Salé-Kénitra standalone {r['target']:10s} best={r['best_variant']:9s} acc={r['best_acc']:.3f} local_maj={r['local_majority']}")
for target, r in zone_comparison.items():
    print(f"  South zone {target:10s} decision={r['decision']} (standalone={r['eddakhla_standalone_acc']}, trio-on-eddakhla={r['trio_acc_on_eddakhla_only']})")
