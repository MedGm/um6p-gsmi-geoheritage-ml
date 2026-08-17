"""
code/30_mixed_effects_leave_region_out.py  (2026-08-17)

Scopes code/24's mixed-effects (region-as-random-intercept) result correctly:
its +3.4pp Difficult win (0.7749 vs deployed ensemble's 0.7408) was measured
under 500m LOGO-cluster CV, where every region is represented in training for
every fold. A random intercept has nothing to borrow from for a region it has
NEVER seen -- for an unseen region it structurally falls back to the
population-average prediction (region_effect=0 in the manual predict formula
below), same mechanism already noted for G2/E2 in this repo. This script runs
the same three logistic-family variants (A_population, B_fixed_dummy,
C_random_intercept) under leave_region_out() instead, mirroring code/20's own
leave_region_out() exactly (same skip-if-n_test<10 rule, same gap_pp/
degenerate reporting), to check whether C's advantage survives true
cross-region generalization or collapses toward A once a region is fully held
out -- exactly the distinction the report already draws for the tree
ensemble ("production predictions are always made in the same-region
setting").

Standardization note: for THIS script (unlike code/24's LOGO-cluster version)
features are standardized using ONLY the training partition (all regions
except the held-out one) for each fold, not the full pooled dataset --
removing an entire region can shift the pooled mean/std non-negligibly,
unlike LOGO-cluster's 1-2-point-per-fold perturbation, so this leakage risk
is worth actually closing here.

Output: results/json/other/mixed_effects_leave_region_out_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

log("Loading N=733 labeled catalog ...")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn: continue
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
region = merged["Region"].values
formula_fe = "y ~ " + " + ".join(FEATURES)

def run_variant_lro(y, variant):
    rows = []
    for reg in pd.unique(region):
        tr_mask = region != reg
        te_mask = region == reg
        n_test = int(te_mask.sum())
        if n_test < 10:
            rows.append({"region": reg, "n_test": n_test, "acc": None, "local_majority": None})
            continue
        Xraw_tr = merged.loc[tr_mask, FEATURES]
        mu, sd = Xraw_tr.mean(), Xraw_tr.std()
        Xs_tr = (Xraw_tr - mu) / sd
        Xs_te = (merged.loc[te_mask, FEATURES] - mu) / sd
        y_tr, y_te = y[tr_mask], y[te_mask]
        if len(np.unique(y_tr)) < 2:
            rows.append({"region": reg, "n_test": n_test, "acc": None, "local_majority": None})
            continue
        try:
            if variant == "A_population":
                m = LogisticRegression(max_iter=1000)
                m.fit(Xs_tr.values, y_tr)
                p = m.predict_proba(Xs_te.values)[:, 1]
            elif variant == "B_fixed_dummy":
                # Manual dummy encoding (not formula/patsy) so an unseen held-out region
                # in the test set gets a well-defined all-zero dummy row (falls back to
                # the reference category) instead of patsy raising on an unknown level --
                # this is the real, well-known failure mode of raw fixed-dummy encoding
                # under true leave-region-out, made explicit rather than crashing.
                train_regions = sorted(pd.unique(merged.loc[tr_mask, "Region"]))
                ref_region = train_regions[0]
                dummy_cols = train_regions[1:]
                region_tr = pd.Categorical(merged.loc[tr_mask, "Region"].values, categories=train_regions)
                region_te = pd.Categorical(merged.loc[te_mask, "Region"].values, categories=train_regions)  # unseen -> NaN -> all-zero dummies
                dtr = pd.get_dummies(region_tr, prefix="Region").reindex(columns=[f"Region_{r}" for r in dummy_cols], fill_value=0)
                dte = pd.get_dummies(region_te, prefix="Region").reindex(columns=[f"Region_{r}" for r in dummy_cols], fill_value=0)
                Xtr_full = pd.concat([Xs_tr.reset_index(drop=True), dtr.reset_index(drop=True)], axis=1)
                Xte_full = pd.concat([Xs_te.reset_index(drop=True), dte.reset_index(drop=True)], axis=1)
                Xtr_full = sm.add_constant(Xtr_full, has_constant="add")
                Xte_full = sm.add_constant(Xte_full, has_constant="add")
                Xte_full = Xte_full.reindex(columns=Xtr_full.columns, fill_value=0)
                res = sm.GLM(y_tr, Xtr_full.astype(float), family=sm.families.Binomial()).fit()
                p = res.predict(Xte_full.astype(float))
            elif variant == "C_random_intercept":
                train = Xs_tr.copy(); train["y"] = y_tr; train["Region"] = pd.Categorical(merged.loc[tr_mask, "Region"].values)
                test = Xs_te.copy(); test["Region"] = merged.loc[te_mask, "Region"].values
                model = BinomialBayesMixedGLM.from_formula(formula_fe, {"Region": "0 + C(Region)"}, train)
                res = model.fit_vb()
                fe = dict(zip(["Intercept"] + FEATURES, res.fe_mean))
                re = res.random_effects()["Mean"].to_dict()
                logit = fe["Intercept"] + sum(fe[f] * test[f].values for f in FEATURES)
                region_eff = np.array([re.get(f"C(Region)[{r}]", 0.0) for r in test["Region"]])  # 0.0 for the held-out region -> population-mean fallback
                p = 1.0 / (1.0 + np.exp(-(logit + region_eff)))
            preds = (p >= 0.5).astype(int)
            acc = float((preds == y_te).mean())
            maj = float(pd.Series(y_te).value_counts(normalize=True).max())
            n_correct = int(round(acc * n_test)); n_correct_maj = int(round(maj * n_test))
            rows.append({"region": reg, "n_test": n_test, "acc": round(acc,3), "local_majority": round(maj,3),
                         "gap_pp": round((acc-maj)*100,1), "degenerate": n_correct == n_correct_maj})
        except Exception as e:
            rows.append({"region": reg, "n_test": n_test, "acc": None, "local_majority": None, "error": str(e)})
    return rows

results = {}
for target_name in ["difficult", "easy"]:
    y = (merged["Expert_Merged"] == ("Difficult" if target_name == "difficult" else "Easy")).astype(int).values
    results[target_name] = {}
    for variant in ["A_population", "B_fixed_dummy", "C_random_intercept"]:
        log(f"Running leave-region-out: {target_name} / {variant} ...")
        rows = run_variant_lro(y, variant)
        results[target_name][variant] = rows
        valid = [r for r in rows if r["acc"] is not None]
        overall_n = sum(r["n_test"] for r in valid)
        if overall_n == 0:
            log(f"  {variant}: NO valid regions (all failed/skipped)")
        else:
            overall_correct = sum(r["acc"] * r["n_test"] for r in valid)
            log(f"  {variant}: pooled-across-regions acc={overall_correct/overall_n:.4f} (n={overall_n})")
        for r in valid:
            print(f"    {r['region']:28s} n={r['n_test']:4d} acc={r['acc']:.3f} maj={r['local_majority']:.3f} gap={r['gap_pp']:+.1f}pp")

out_path = os.path.join(BASE, "results", "json", "other", "mixed_effects_leave_region_out_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log(f"Wrote {out_path}")
