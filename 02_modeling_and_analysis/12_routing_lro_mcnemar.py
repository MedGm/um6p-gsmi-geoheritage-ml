"""
data_audit/12_routing_lro_mcnemar.py  (2026-08-22)

Fixes a mistake in 11_routing_mcnemar_regional.py: that script paired
LOGO-cluster CV predictions (which still train on other clusters WITHIN
TTAH/Souss-Massa), but the apparent regional improvement was reported under
leave-region-out (zero exposure to the held-out region during training) --
a different, harder protocol. Testing the wrong pairing gave a null result
that doesn't actually speak to the real claim. This redoes it correctly:
per-site leave-region-out predictions, for both Baseline_939_difficult and
RoutingReplace_939_difficult, restricted to Tanger-Tétouan-Al Hoceima and
Souss-Massa, paired McNemar on those.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
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

def lro_predict_for_regions(X, y, region, best_cfgs, target_regions):
    """Per-site leave-region-out predictions, only for the given target regions."""
    preds = {}
    for reg in target_regions:
        tr = region != reg; te = region == reg
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

TARGET_REGIONS = ["Tanger-Tétouan-Al Hoceima", "Souss-Massa"]

log("Leave-region-out predictions: Baseline_939_difficult, TTAH + Souss-Massa ...")
X_base = merged[FEATURES_BASE].values
preds_base = lro_predict_for_regions(X_base, y, region, baseline_cfgs, TARGET_REGIONS)

log("Leave-region-out predictions: RoutingReplace_939_difficult, TTAH + Souss-Massa ...")
X_routing = merged[FEATURES_ROUTING].values
preds_routing = lro_predict_for_regions(X_routing, y, region, routing_cfgs, TARGET_REGIONS)

idx = sorted(preds_base.keys())
assert idx == sorted(preds_routing.keys())
y_sub = y[idx]
pred_base = np.array([preds_base[i] for i in idx])
pred_routing = np.array([preds_routing[i] for i in idx])
correct_base = pred_base == y_sub
correct_routing = pred_routing == y_sub

acc_base = correct_base.mean()
acc_routing = correct_routing.mean()
log(f"N={len(idx)} (TTAH+Souss-Massa, leave-region-out) acc_baseline={acc_base:.4f} acc_routing={acc_routing:.4f}")

table = pd.crosstab(pd.Series(correct_base), pd.Series(correct_routing)).reindex(
    index=[False, True], columns=[False, True], fill_value=0).values
res = mcnemar(table, exact=False, correction=True)
n10 = int(((correct_base == True) & (correct_routing == False)).sum())
n01 = int(((correct_base == False) & (correct_routing == True)).sum())
log(f"McNemar (leave-region-out, TTAH+Souss-Massa): chi2={res.statistic:.4f} p={res.pvalue:.4f} "
    f"(baseline_right_routing_wrong={n10}, baseline_wrong_routing_right={n01})")

# also per-region breakdown for sanity vs the originally-reported gap_pp numbers
for reg in TARGET_REGIONS:
    mask = region[idx] == reg
    n = mask.sum()
    ab = correct_base[mask].mean(); ar = correct_routing[mask].mean()
    log(f"  {reg}: N={n} acc_baseline={ab:.3f} acc_routing={ar:.3f}")
