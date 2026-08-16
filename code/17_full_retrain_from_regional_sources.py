"""
Full retrain from the current, complete label set: reads every regional_label_sources/
CSV fresh (not the stale N=260 combined_expert_labels.csv, not the separate
labeling_candidates staging files) -- this is the single, always-current source of
truth as of 2026-08-12: N=344 unique labeled sites (251 original + 9 previously-
excluded BMK + 84 new labels added today across Drâa-Tafilalet/BMK/Dakhla/Rabat-Salé-
Kénitra). Same cluster-aware methodology as code/09 (500m haversine locality
clustering, StratifiedGroupKFold tuning, LeaveOneGroupOut final evaluation) so the
result is directly comparable to the original N=251 baseline -- no confound, no
mislabeled baseline, no silently-dropped rows (all bugs found 2026-08-12 in
code/15 and code/16 are avoided here by construction, not patched).

Run (lid/sleep-safe, background):
    cd geosite_project1
    source venv/bin/activate   # if using a venv
    nohup systemd-inhibit --what=sleep:idle --why="ML retrain" \\
      python3 code/17_full_retrain_from_regional_sources.py > retrain_full.log 2>&1 &
    echo "PID: $!"

Watch progress:
    tail -f retrain_full.log

Expect ~2-3 hours on an 8-thread laptop (this grid is larger than code/09's, N=344
vs N=251, similar per-fit cost). Do NOT use n_jobs=-1 per-fit (caused a 4h+ stall on
the Z8 workstation before, see project memory) -- this script already fixes that
(n_jobs=1 throughout, single-threaded fits, since N is small enough that pool-spawn
overhead would dominate over the actual compute at this scale).
"""
import glob, json, os, time, warnings
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
t0 = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
MOUT = os.path.join(BASE, "data", "model_outputs")
MMOD = os.path.join(BASE, "models")
os.makedirs(MOUT, exist_ok=True)
os.makedirs(MMOD, exist_ok=True)

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

# N=251 baseline from code/09 (2026-08-09) -- the ONLY prior number this is safely
# comparable to, since N=344 here is a clean superset (251 + 9 old BMK + 84 new),
# not confounded by any silent baseline mismatch.
BASELINE = {
    "3class":    0.5776892430278885,
    "difficult": 0.7928286852589641,
    "easy":      0.8207171314741036,
}

# ── 1. Load ALL labels fresh from regional_label_sources/*.csv ─────────────────
print("=== 1. Loading labels from regional_label_sources/*.csv ===", flush=True)
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data", "final", "regional_label_sources", "*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
    print(f"  {os.path.basename(f)}: {len(labeled)} labeled", flush=True)
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
print(f"  Total unique labeled: {len(all_labels)}", flush=True)

# ── 2. Merge with features from the (clean, 19-col) main catalog ───────────────
catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
n_dropped = len(all_labels) - len(merged)
if n_dropped:
    missing = set(all_labels["Locality_ID"]) - set(merged["Locality_ID"])
    print(f"  WARNING: {n_dropped} labeled row(s) had no matching features: {missing}", flush=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
print(f"\n=== Merged N={N} ===", flush=True)
print(f"  3-class: {merged['Expert_Merged'].value_counts().to_dict()}", flush=True)

# ── 3. Locality clustering (500m haversine union-find, same as code/09) ────────
print("\n=== 2. Cluster assignment ===", flush=True)
lat, lon = merged["Latitude_WGS84"].values, merged["Longitude_WGS84"].values
n = len(merged)
def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lr[:, None]) * np.cos(lr[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
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
    for j in range(i + 1, n):
        if D[i, j] <= 500: union(i, j)
cluster_ids = np.array([find(i) for i in range(n)])
n_clusters = len(np.unique(cluster_ids))
print(f"  N={n}, clusters={n_clusters}", flush=True)

X = merged[FEATURES].values
logo = LeaveOneGroupOut()

# ── 4. Grid + eval helpers (n_jobs=1 throughout -- see module docstring) ───────
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
    for ne in [100, 200, 400]:
        for md in [3, 4, 5, 6, 7, 8, None]:
            for msl in [1, 2, 3, 5]:
                f = lambda ne=ne, md=md, msl=msl: RandomForestClassifier(
                    n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                    class_weight="balanced", random_state=42, n_jobs=1)
                m, s = cv_score(f, X, y, cluster_ids)
                results.append({"model": "RF", "n_est": ne, "max_depth": md, "min_leaf": msl, "mean_acc": m, "std_acc": s})
    print(f"[{tag}] RF done [{time.time()-t0:.0f}s]", flush=True)
    for ne in [100, 150, 250]:
        for md in [3, 4, 5, 6]:
            for lr in [0.03, 0.08, 0.15]:
                f = lambda ne=ne, md=md, lr=lr: XGBClassifier(
                    n_estimators=ne, max_depth=md, learning_rate=lr, random_state=42, n_jobs=1,
                    eval_metric="mlogloss" if tag == "3class" else "logloss")
                m, s = cv_score(f, X, y, cluster_ids)
                results.append({"model": "XGB", "n_est": ne, "max_depth": md, "lr": lr, "mean_acc": m, "std_acc": s})
    print(f"[{tag}] XGB done [{time.time()-t0:.0f}s]", flush=True)
    for ne in [100, 200]:
        for md in [2, 3, 4]:
            for lr in [0.05, 0.1]:
                f = lambda ne=ne, md=md, lr=lr: GradientBoostingClassifier(
                    n_estimators=ne, max_depth=md, learning_rate=lr, random_state=42)
                m, s = cv_score(f, X, y, cluster_ids)
                results.append({"model": "GBM", "n_est": ne, "max_depth": md, "lr": lr, "mean_acc": m, "std_acc": s})
    print(f"[{tag}] GBM done [{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(results).sort_values("mean_acc", ascending=False)

def build_and_eval(df_res, y, tag, class_names):
    top_rf = df_res[df_res["model"] == "RF"].iloc[0]
    top_xgb = df_res[df_res["model"] == "XGB"].iloc[0]
    top_gbm = df_res[df_res["model"] == "GBM"].iloc[0]
    rf_f = RandomForestClassifier(
        n_estimators=int(top_rf["n_est"]),
        max_depth=None if pd.isna(top_rf["max_depth"]) else int(top_rf["max_depth"]),
        min_samples_leaf=int(top_rf["min_leaf"]), class_weight="balanced", random_state=42, n_jobs=1)
    xgb_f = XGBClassifier(
        n_estimators=int(top_xgb["n_est"]), max_depth=int(top_xgb["max_depth"]),
        learning_rate=top_xgb["lr"], random_state=42, n_jobs=1,
        eval_metric="mlogloss" if tag == "3class" else "logloss")
    gbm_f = GradientBoostingClassifier(
        n_estimators=int(top_gbm["n_est"]), max_depth=int(top_gbm["max_depth"]), learning_rate=top_gbm["lr"], random_state=42)
    preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups=cluster_ids):
        m = VotingClassifier(estimators=[("rf", rf_f), ("xgb", xgb_f), ("gbm", gbm_f)], voting="soft", n_jobs=1)
        m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    acc = accuracy_score(y, preds)
    print(f"[{tag}] Ensemble LOGO-cluster CV: {acc:.4f} ({int(acc*len(y))}/{len(y)})  [{time.time()-t0:.0f}s]", flush=True)
    print(confusion_matrix(y, preds), flush=True)
    print(classification_report(y, preds, target_names=class_names, zero_division=0), flush=True)
    ens = VotingClassifier(estimators=[("rf", rf_f), ("xgb", xgb_f), ("gbm", gbm_f)], voting="soft", n_jobs=1)
    ens.fit(X, y)
    summary = {"N": N, "n_clusters": int(n_clusters), "best_rf": top_rf.to_dict(),
               "best_xgb": top_xgb.to_dict(), "best_gbm": top_gbm.to_dict(),
               "cluster_holdout_acc": acc, "baseline_acc_N251": BASELINE.get(tag),
               "delta_pp": round((acc - BASELINE.get(tag, 0)) * 100, 2)}
    return ens, acc, summary

# ── 5. Run all three targets ────────────────────────────────────────────────────
print("\n=== 3-class target ===", flush=True)
y3 = merged["Expert_Merged"].map({"Easy": 0, "Moderate": 1, "Difficult": 2}).values
df3 = run_grid(y3, "3class")
df3.to_csv(os.path.join(MOUT, f"hyperparameter_search_results_N{N}.csv"), index=False)
ens3, acc3, summ3 = build_and_eval(df3, y3, "3class", ["Easy", "Moderate", "Difficult"])
joblib.dump(ens3, os.path.join(MMOD, f"national_pilot_model_N{N}.joblib"))
json.dump(summ3, open(os.path.join(MOUT, f"hyperparameter_search_summary_N{N}.json"), "w"), indent=2, default=str)

print("\n=== Binary Difficult ===", flush=True)
yd = (merged["Expert_Merged"] == "Difficult").astype(int).values
dfd = run_grid(yd, "difficult")
dfd.to_csv(os.path.join(MOUT, f"binary_hyperparameter_search_results_N{N}.csv"), index=False)
ensd, accd, summd = build_and_eval(dfd, yd, "difficult", ["Not-Difficult", "Difficult"])
joblib.dump(ensd, os.path.join(MMOD, f"national_binary_model_N{N}.joblib"))
json.dump(summd, open(os.path.join(MOUT, f"binary_hyperparameter_search_summary_N{N}.json"), "w"), indent=2, default=str)

print("\n=== Binary Easy ===", flush=True)
ye = (merged["Expert_Merged"] == "Easy").astype(int).values
dfe = run_grid(ye, "easy")
dfe.to_csv(os.path.join(MOUT, f"easy_hyperparameter_search_results_N{N}.csv"), index=False)
ense, acce, summe = build_and_eval(dfe, ye, "easy", ["Not-Easy", "Easy"])
joblib.dump(ense, os.path.join(MMOD, f"national_easy_model_N{N}.joblib"))
json.dump(summe, open(os.path.join(MOUT, f"easy_hyperparameter_search_summary_N{N}.json"), "w"), indent=2, default=str)

# ── 6. Final comparison ─────────────────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print(f"  COMPARISON: N=251 baseline (2026-08-09) vs N={N} (2026-08-12, clean superset)", flush=True)
print("=" * 65, flush=True)
print(f"{'Target':<22}{'N=251':>10}{'N='+str(N):>10}{'Delta':>10}", flush=True)
print("-" * 65, flush=True)
for tag, base, new_acc in [("3-class", BASELINE["3class"], acc3),
                            ("Binary Difficult", BASELINE["difficult"], accd),
                            ("Binary Easy", BASELINE["easy"], acce)]:
    print(f"{tag:<22}{base*100:>9.1f}%{new_acc*100:>9.1f}%{(new_acc-base)*100:>+9.1f}pp", flush=True)
print("=" * 65, flush=True)
print(f"N={N}  |  clusters={n_clusters}  |  elapsed={time.time()-t0:.0f}s", flush=True)
print("DONE.", flush=True)
