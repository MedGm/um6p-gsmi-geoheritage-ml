"""
data_audit/13_routing_lro_all_regions.py  (2026-08-22)

Follow-up to 12_routing_lro_mcnemar.py, which found a significant leave-
region-out improvement from the routing-distance feature (RoutingReplace_939
vs Baseline_939, Difficult target) but only tested it on Tanger-Tétouan-Al
Hoceima + Souss-Massa -- the two regions that looked promising in the
earlier exploratory regional breakdown. That's a selected-after-seeing-the-
data risk, not yet a general finding. This reruns leave-region-out
per-site predictions across EVERY region with enough test data (same >=10
threshold as the original leave_region_out()), for both feature sets, then
reports per-region accuracy AND a pooled McNemar across all regions
together -- either the effect replicates broadly (real, general
cross-region-generalization gain) or it's specific to TTAH/Souss-Massa's
terrain (still real, just narrower than a first read might suggest).
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
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
FEATURES_ROUTING = ["Dist_to_Highway_Routing_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

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
    catalog[["Locality_ID", "Region"] + FEATURES_BASE], on="Locality_ID", how="inner"
).merge(routing, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662
y = (merged["Expert_Merged"] == "Difficult").astype(int).values
region = merged["Region"].values

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def lro_predict_all(X, y, region, best_cfgs, min_test_n=10):
    preds = {}
    for reg in pd.unique(region):
        tr = region != reg; te = region == reg
        if te.sum() < min_test_n:
            continue
        sw = combined_weight(y[tr])
        estimators = [(k.lower(), make_model(k, cfg)) for k, cfg in best_cfgs]
        m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        m.fit(X[tr], y[tr], sample_weight=sw)
        p = m.predict(X[te])
        idx = np.where(te)[0]
        for i, pred in zip(idx, p):
            preds[i] = pred
    return preds

modeling_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_modeling_results.json")))
routing_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_routing_feature_results.json")))
baseline_cfgs = modeling_results["Baseline_939_difficult"]["best_configs"]
routing_cfgs = routing_results["RoutingReplace_939_difficult"]["best_configs"]

log("Leave-region-out predictions: Baseline_939_difficult, ALL qualifying regions ...")
X_base = merged[FEATURES_BASE].values
preds_base = lro_predict_all(X_base, y, region, baseline_cfgs)

log("Leave-region-out predictions: RoutingReplace_939_difficult, ALL qualifying regions ...")
X_routing = merged[FEATURES_ROUTING].values
preds_routing = lro_predict_all(X_routing, y, region, routing_cfgs)

idx = sorted(set(preds_base.keys()) & set(preds_routing.keys()))
y_sub = y[idx]
reg_sub = region[idx]
pred_base = np.array([preds_base[i] for i in idx])
pred_routing = np.array([preds_routing[i] for i in idx])
correct_base = pred_base == y_sub
correct_routing = pred_routing == y_sub

log(f"\nN={len(idx)} total across all qualifying regions (leave-region-out)")
log(f"Overall acc_baseline={correct_base.mean():.4f} acc_routing={correct_routing.mean():.4f}")

table = pd.crosstab(pd.Series(correct_base), pd.Series(correct_routing)).reindex(
    index=[False, True], columns=[False, True], fill_value=0).values
res = mcnemar(table, exact=False, correction=True)
n10 = int(((correct_base == True) & (correct_routing == False)).sum())
n01 = int(((correct_base == False) & (correct_routing == True)).sum())
log(f"POOLED McNemar (all regions, leave-region-out): chi2={res.statistic:.4f} p={res.pvalue:.4f} "
    f"(baseline_right_routing_wrong={n10}, baseline_wrong_routing_right={n01})")

log("\nPer-region breakdown:")
rows = []
for reg in pd.unique(reg_sub):
    mask = reg_sub == reg
    n = mask.sum()
    ab, ar = correct_base[mask].mean(), correct_routing[mask].mean()
    delta_pp = (ar - ab) * 100
    rows.append((reg, n, ab, ar, delta_pp))
rows.sort(key=lambda r: -r[4])
for reg, n, ab, ar, delta_pp in rows:
    log(f"  {reg:28s} N={n:4d} baseline={ab:.3f} routing={ar:.3f} delta={delta_pp:+.1f}pp")
