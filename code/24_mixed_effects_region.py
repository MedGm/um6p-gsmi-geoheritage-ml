"""
code/24_mixed_effects_region.py  (2026-08-17)

Tests whether a proper random-intercept (mixed-effects) treatment of Region
beats the raw region-as-fixed-dummy encoding already tried in code/18's E2 /
code/20's G2 (acc_logo_cluster=0.6235, but that was the 3-class model; no
binary-decomposition region-as-feature comparison exists yet in this repo).

Isolates the REGION-HANDLING MECHANISM from model family: all three variants
below are logistic-family GLMs on the same 6 standardized base features, same
500m-cluster LeaveOneGroupOut CV protocol as code/20 (cluster_of()). The tree-
ensemble headline numbers are a separate question (already addressed by the
model-family comparison in make_figures.py); this experiment only asks
whether adding region information, and if so, whether raw dummy encoding or
partial-pooling shrinkage, helps GENERALIZATION WITHIN the observed regions
(LOGO-cluster CV -- region is never fully held out, matching G2's own
methodology note: "leave-region-out is not meaningful for it").

Variants, per binary target (Difficult-vs-not, Easy-vs-not):
  A. Population-only logistic  (no region term)             -- fixed-effect baseline
  B. Region-as-fixed-dummy     (C(Region) dummies, statsmodels GLM Binomial)
  C. Region-as-random-intercept (BinomialBayesMixedGLM, region as variance component)
     -- partial pooling: small regions borrow strength from the population
     mean instead of getting a free, unregularized dummy coefficient. For a
     region unseen in a training fold (should not occur under LOGO-cluster
     CV, only under true leave-region-out), C's prediction degrades exactly
     to A's (population mean, zero random effect) -- the structural
     limitation flagged in the day's research-plan discussion.

All three are UNWEIGHTED (no confidence/class-balance sample weights) to keep
the comparison to the region-handling mechanism alone; weighting is already
tested elsewhere (code/20) as a separate, orthogonal lever.

Standardization note: features are z-scored ONCE on the full pooled dataset
(not per-fold) -- a documented simplification. With 622 LOGO-cluster folds
each removing only 1-2 points, the perturbation to the global mean/std is
negligible; this is not the region-relative z-score leakage mechanism
identified for F4 (that leaked REGION IDENTITY via per-region normalization
constants -- this uses one pooled constant for everyone).

Output: results/json/other/mixed_effects_region_results.json
"""
import glob, json, os, time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

log("Loading N=733 labeled catalog ...")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 733
log(f"N={N}")

Xraw = merged[FEATURES].copy()
Xs = (Xraw - Xraw.mean()) / Xraw.std()
data = Xs.copy()
data["Region"] = pd.Categorical(merged["Region"].values)
regions_all = list(data["Region"].cat.categories)
log(f"regions: {len(regions_all)}")

def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cluster_of(sub, radius_m=500):
    lat, lon = sub["Latitude_WGS84"].values, sub["Longitude_WGS84"].values
    n = len(sub)
    D = haversine_matrix(lat, lon)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] <= radius_m:
                rx, ry = find(i), find(j)
                if rx != ry: parent[rx] = ry
    return np.array([find(i) for i in range(n)])

cluster_ids = cluster_of(merged)
n_clusters = len(np.unique(cluster_ids))
log(f"clusters={n_clusters} (singleton rate {(pd.Series(cluster_ids).value_counts()==1).mean()*100:.1f}%)")

formula_fe = "y ~ " + " + ".join(FEATURES)
formula_dummy = "y ~ " + " + ".join(FEATURES) + " + C(Region)"
fe_col_order = ["Intercept"] + FEATURES

def run_variant(target_col, variant):
    logo = LeaveOneGroupOut()
    preds = np.full(N, np.nan)
    n_fail = 0
    for train_idx, test_idx in logo.split(data, groups=cluster_ids):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        train["y"] = target_col[train_idx]
        if train["y"].nunique() < 2:
            continue  # degenerate fold, can't fit -- leave as nan, excluded from accuracy
        try:
            if variant == "A_population":
                mdl = LogisticRegression(max_iter=1000)
                mdl.fit(train[FEATURES].values, train["y"].values)
                p = mdl.predict_proba(test[FEATURES].values)[:, 1]
            elif variant == "B_fixed_dummy":
                res = smf.glm(formula_dummy, data=train, family=sm.families.Binomial()).fit()
                p = res.predict(test)
            elif variant == "C_random_intercept":
                model = BinomialBayesMixedGLM.from_formula(
                    formula_fe, {"Region": "0 + C(Region)"}, train)
                res = model.fit_vb()
                fe = dict(zip(["Intercept"] + FEATURES, res.fe_mean))
                re = res.random_effects()["Mean"].to_dict()
                logit = fe["Intercept"] + sum(fe[f] * test[f].values for f in FEATURES)
                region_eff = np.array([re.get(f"C(Region)[{r}]", 0.0) for r in test["Region"]])
                logit = logit + region_eff
                p = 1.0 / (1.0 + np.exp(-logit))
            preds[test_idx] = p
        except Exception:
            n_fail += 1
            continue
    return preds, n_fail

results = {}
for target_name, target_expr in [("difficult", merged["Expert_Merged"] == "Difficult"),
                                   ("easy", merged["Expert_Merged"] == "Easy")]:
    y = target_expr.astype(int).values
    results[target_name] = {}
    for variant in ["A_population", "B_fixed_dummy", "C_random_intercept"]:
        log(f"Running {target_name} / {variant} ({n_clusters} folds) ...")
        preds, n_fail = run_variant(y, variant)
        valid = ~np.isnan(preds)
        acc = float(((preds[valid] >= 0.5).astype(int) == y[valid]).mean())
        results[target_name][variant] = {
            "acc_logo_cluster": round(acc, 4),
            "n_valid": int(valid.sum()),
            "n_failed_folds": n_fail,
        }
        log(f"  {target_name}/{variant}: acc={acc:.4f} (n_valid={valid.sum()}, failed={n_fail})")

out_path = os.path.join(BASE, "results", "json", "other", "mixed_effects_region_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log(f"Wrote {out_path}")

log("Summary:")
for target_name in ["difficult", "easy"]:
    print(f"  {target_name}:")
    for variant in ["A_population", "B_fixed_dummy", "C_random_intercept"]:
        r = results[target_name][variant]
        print(f"    {variant:22s} acc={r['acc_logo_cluster']:.4f}")
