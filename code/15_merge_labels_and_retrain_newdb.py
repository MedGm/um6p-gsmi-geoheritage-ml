"""
Meeting-4 follow-up (4 Aug 2026): merge 49 newly-labeled newdb_v2 sites
(BMK + Draa-Tafilalet pilots) with the existing 260-label ground truth and
retrain all three ensemble models with cluster-aware CV (same methodology
as script 09). Prints side-by-side comparison vs N=260 baseline.

Outputs
-------
data/newdb_v2/labeling_candidates/combined_newdb_labels.csv
data/model_outputs/hyperparameter_search_results_N308.csv + _summary_N308.json
data/model_outputs/binary_hyperparameter_search_results_N308.csv + _summary_N308.json
data/model_outputs/easy_hyperparameter_search_results_N308.csv + _summary_N308.json
models/national_pilot_model_N308.joblib
models/national_binary_model_N308.joblib
models/national_easy_model_N308.joblib
"""
import json, os, time, warnings
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier, VotingClassifier)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
t0 = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))

# N=251 cluster-aware baselines from script 09 (NOT N=260 -- corrected 2026-08-12).
# The "old" data merged below (data/final/combined_expert_labels.csv) has 260 rows,
# including 9 post-report-addition BMK sites that script 09 deliberately EXCLUDED to
# reconstruct the official N=251 evaluation set. So this baseline and the N=308 merge
# differ by TWO things at once (the 9 reintroduced BMK sites, and the 48 new labels) --
# any delta printed below is confounded, not a clean "did the new labels help" number.
# A true apples-to-apples 3-way comparison (N=251 -> N=260-only -> N=308) needs a
# separate N=260-only run, not done here (expensive, not launched automatically).
BASELINE = {
    "3class":    0.5776892430278885,
    "difficult": 0.7928286852589641,
    "easy":      0.8207171314741036,
}

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

# ── 1. Old labels + features ────────────────────────────────────────────────
print("=== 1. Old labels + features ===", flush=True)
old_labels = pd.read_csv(os.path.join(BASE, "data", "final", "combined_expert_labels.csv"))
old_feats  = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
old = old_labels.merge(
    old_feats[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
old["Source"] = "old_final_260"
print(f"  Old labeled+features: {len(old)}", flush=True)

# ── 2. New pilot labels + features ──────────────────────────────────────────
print("=== 2. New pilot labels + features ===", flush=True)
LAB_DIR = os.path.join(BASE, "data", "newdb_v2", "labeling_candidates")
bmk = pd.read_csv(os.path.join(LAB_DIR, "bmk_pilot_results.csv"))
dt  = pd.read_csv(os.path.join(LAB_DIR, "draa_tafilalet_pilot_results.csv"))
new_labels_raw = pd.concat([
    bmk[bmk["Expert_Class"].notna()][
        ["Locality_ID","Geosite_Name","Expert_Class","Confidence","Expert_Reasoning"]
    ].assign(Source="bmk_pilot"),
    dt[dt["Expert_Class"].notna()][
        ["Locality_ID","Geosite_Name","Expert_Class","Confidence","Expert_Reasoning"]
    ].assign(Source="dt_pilot"),
], ignore_index=True)
# Source new-locality features from the authoritative main catalog, not the
# geosites_new_localities_features.csv intermediate staging file -- confirmed
# 2026-08-12 that file is stale (missing loc_00775 despite it having complete, valid
# features in the main catalog), which silently dropped 1 of 49 labeled rows with no
# warning. The main catalog has all 1154 sites (old + new) with the 6 features intact.
new_feats = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
new = new_labels_raw.merge(
    new_feats[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
n_labeled_raw = len(new_labels_raw)
if len(new) != n_labeled_raw:
    missing = set(new_labels_raw["Locality_ID"]) - set(new["Locality_ID"])
    print(f"  WARNING: {n_labeled_raw - len(new)} labeled row(s) dropped (no matching "
          f"features): {missing}", flush=True)
print(f"  New labeled+features: {len(new)}", flush=True)
print(f"  Classes: {new['Expert_Class'].value_counts().to_dict()}", flush=True)

# ── 3. Overlap check ────────────────────────────────────────────────────────
overlap = set(old["Locality_ID"]) & set(new["Locality_ID"])
assert len(overlap) == 0, f"ID overlap: {overlap}"
print(f"  Overlap: 0  ✓", flush=True)

# ── 4. Save audit file ──────────────────────────────────────────────────────
old_audit = old[["Locality_ID","Expert_Class"]].copy()
old_audit["Geosite_Name"] = (old_feats.set_index("Locality_ID")
                              .reindex(old["Locality_ID"])["Geosite_Name"].values)
old_audit["Confidence"] = "from_old_pipeline"
old_audit["Expert_Reasoning"] = "see data/final/regional_label_sources/"
old_audit["Source"] = "old_final_260"
audit = pd.concat([old_audit,
                   new[["Locality_ID","Geosite_Name","Expert_Class",
                         "Confidence","Expert_Reasoning","Source"]]],
                  ignore_index=True)
audit.to_csv(os.path.join(LAB_DIR, "combined_newdb_labels.csv"), index=False)
print(f"  Audit saved ({len(audit)} rows)", flush=True)

# ── 5. Merge ─────────────────────────────────────────────────────────────────
merged = pd.concat([
    old[["Locality_ID","Latitude_WGS84","Longitude_WGS84","Expert_Class"] + FEATURES],
    new[["Locality_ID","Latitude_WGS84","Longitude_WGS84","Expert_Class"] + FEATURES],
], ignore_index=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
print(f"\n=== Merged N={N} ===", flush=True)
print(f"  3-class: {merged['Expert_Merged'].value_counts().to_dict()}", flush=True)

# ── 6. Cluster assignment (500m haversine union-find, same as script 09) ────
print("\n=== 6. Cluster assignment ===", flush=True)
lat = merged["Latitude_WGS84"].values
lon = merged["Longitude_WGS84"].values
n = len(merged)

def haversine_matrix(lat, lon):
    R = 6371000
    lr = np.radians(lat); lo = np.radians(lon)
    dlat = lr[:,None]-lr[None,:]; dlon = lo[:,None]-lo[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

D = haversine_matrix(lat, lon)
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry
for i in range(n):
    for j in range(i+1, n):
        if D[i,j] <= 500: union(i,j)
cluster_ids = np.array([find(i) for i in range(n)])
n_clusters = len(np.unique(cluster_ids))
print(f"  N={n}, clusters={n_clusters}", flush=True)

X = merged[FEATURES].values
logo = LeaveOneGroupOut()

# ── 7. Grid + evaluation helpers ─────────────────────────────────────────────
def cv_score(factory, X, y, groups, n_repeats=5, n_splits=10):
    scores = []
    for rep in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, y, groups=groups):
            m = factory(); m.fit(X[tr], y[tr])
            scores.append(accuracy_score(y[te], m.predict(X[te])))
    return np.mean(scores), np.std(scores)

def run_grid(y, tag):
    results = []
    for ne in [100,200,400]:
        for md in [3,4,5,6,7,8,None]:
            for msl in [1,2,3,5]:
                f = lambda ne=ne,md=md,msl=msl: RandomForestClassifier(
                    n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                    class_weight="balanced", random_state=42, n_jobs=1)
                m,s = cv_score(f, X, y, cluster_ids)
                results.append({"model":"RF","n_est":ne,"max_depth":md,"min_leaf":msl,
                                 "mean_acc":m,"std_acc":s})
    print(f"[{tag}] RF done [{time.time()-t0:.0f}s]", flush=True)
    for ne in [100,150,250]:
        for md in [3,4,5,6]:
            for lr in [0.03,0.08,0.15]:
                f = lambda ne=ne,md=md,lr=lr: XGBClassifier(
                    n_estimators=ne, max_depth=md, learning_rate=lr,
                    random_state=42, n_jobs=1,
                    eval_metric="mlogloss" if tag=="3class" else "logloss")
                m,s = cv_score(f, X, y, cluster_ids)
                results.append({"model":"XGB","n_est":ne,"max_depth":md,"lr":lr,
                                 "mean_acc":m,"std_acc":s})
    print(f"[{tag}] XGB done [{time.time()-t0:.0f}s]", flush=True)
    for ne in [100,200]:
        for md in [2,3,4]:
            for lr in [0.05,0.1]:
                f = lambda ne=ne,md=md,lr=lr: GradientBoostingClassifier(
                    n_estimators=ne, max_depth=md, learning_rate=lr, random_state=42)
                m,s = cv_score(f, X, y, cluster_ids)
                results.append({"model":"GBM","n_est":ne,"max_depth":md,"lr":lr,
                                 "mean_acc":m,"std_acc":s})
    print(f"[{tag}] GBM done [{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(results).sort_values("mean_acc", ascending=False)

def build_and_eval(df_res, y, tag, class_names):
    top_rf  = df_res[df_res["model"]=="RF"].iloc[0]
    top_xgb = df_res[df_res["model"]=="XGB"].iloc[0]
    top_gbm = df_res[df_res["model"]=="GBM"].iloc[0]
    rf_f = RandomForestClassifier(
        n_estimators=int(top_rf["n_est"]),
        max_depth=None if pd.isna(top_rf["max_depth"]) else int(top_rf["max_depth"]),
        min_samples_leaf=int(top_rf["min_leaf"]),
        class_weight="balanced", random_state=42, n_jobs=1)
    xgb_f = XGBClassifier(
        n_estimators=int(top_xgb["n_est"]), max_depth=int(top_xgb["max_depth"]),
        learning_rate=top_xgb["lr"], random_state=42, n_jobs=1,
        eval_metric="mlogloss" if tag=="3class" else "logloss")
    gbm_f = GradientBoostingClassifier(
        n_estimators=int(top_gbm["n_est"]), max_depth=int(top_gbm["max_depth"]),
        learning_rate=top_gbm["lr"], random_state=42)
    preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups=cluster_ids):
        m = VotingClassifier(estimators=[("rf",rf_f),("xgb",xgb_f),("gbm",gbm_f)],
                              voting="soft", n_jobs=1)
        m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    acc = accuracy_score(y, preds)
    print(f"[{tag}] Ensemble LOGO-cluster CV: {acc:.4f} "
          f"({int(acc*len(y))}/{len(y)})  [{time.time()-t0:.0f}s]", flush=True)
    print(confusion_matrix(y, preds), flush=True)
    print(classification_report(y, preds, target_names=class_names), flush=True)
    ens = VotingClassifier(estimators=[("rf",rf_f),("xgb",xgb_f),("gbm",gbm_f)],
                            voting="soft", n_jobs=1)
    ens.fit(X, y)
    summary = {"N": N, "n_clusters": int(n_clusters),
               "best_rf": top_rf.to_dict(), "best_xgb": top_xgb.to_dict(),
               "best_gbm": top_gbm.to_dict(), "cluster_holdout_acc": acc,
               "baseline_acc_N260": BASELINE.get(tag),
               "delta_pp": round((acc - BASELINE.get(tag, 0))*100, 2)}
    return ens, acc, summary

# ── 8. Run all three targets ──────────────────────────────────────────────────
MOUT = os.path.join(BASE, "data", "model_outputs")
MMOD = os.path.join(BASE, "models")

print("\n=== 3-class target ===", flush=True)
y3 = merged["Expert_Merged"].map({"Easy":0,"Moderate":1,"Difficult":2}).values
df3 = run_grid(y3, "3class")
df3.to_csv(os.path.join(MOUT, "hyperparameter_search_results_N308.csv"), index=False)
ens3, acc3, summ3 = build_and_eval(df3, y3, "3class", ["Easy","Moderate","Difficult"])
joblib.dump(ens3, os.path.join(MMOD, "national_pilot_model_N308.joblib"))
json.dump(summ3, open(os.path.join(MOUT, "hyperparameter_search_summary_N308.json"),"w"),
          indent=2, default=str)

print("\n=== Binary Difficult ===", flush=True)
yd = (merged["Expert_Merged"]=="Difficult").astype(int).values
dfd = run_grid(yd, "difficult")
dfd.to_csv(os.path.join(MOUT, "binary_hyperparameter_search_results_N308.csv"), index=False)
ensd, accd, summd = build_and_eval(dfd, yd, "difficult", ["Not-Difficult","Difficult"])
joblib.dump(ensd, os.path.join(MMOD, "national_binary_model_N308.joblib"))
json.dump(summd, open(os.path.join(MOUT, "binary_hyperparameter_search_summary_N308.json"),"w"),
          indent=2, default=str)

print("\n=== Binary Easy ===", flush=True)
ye = (merged["Expert_Merged"]=="Easy").astype(int).values
dfe = run_grid(ye, "easy")
dfe.to_csv(os.path.join(MOUT, "easy_hyperparameter_search_results_N308.csv"), index=False)
ense, acce, summe = build_and_eval(dfe, ye, "easy", ["Not-Easy","Easy"])
joblib.dump(ense, os.path.join(MMOD, "national_easy_model_N308.joblib"))
json.dump(summe, open(os.path.join(MOUT, "easy_hyperparameter_search_summary_N308.json"),"w"),
          indent=2, default=str)

# ── 9. Final comparison table ─────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  COMPARISON: cluster-aware LOGO CV (Very Difficult → Difficult)", flush=True)
print("="*65, flush=True)
print(f"{'Target':<22}{'N=260 baseline':>16}{'N=308 new':>12}{'Delta':>10}", flush=True)
print("-"*65, flush=True)
for tag, base, new_acc, lbl in [
    ("3-class",          BASELINE["3class"],    acc3,  "3class"),
    ("Binary Difficult", BASELINE["difficult"], accd,  "difficult"),
    ("Binary Easy",      BASELINE["easy"],      acce,  "easy"),
]:
    print(f"{tag:<22}{base*100:>15.1f}%{new_acc*100:>11.1f}%"
          f"{(new_acc-base)*100:>+9.1f}pp", flush=True)
print("="*65, flush=True)
print(f"N merged={N}  |  clusters={n_clusters}  |  "
      f"elapsed={time.time()-t0:.0f}s", flush=True)
print("DONE.", flush=True)
