"""
Incremental retrain: after Phase-2 labeling batches are filled in, this script
reads all completed pilot result CSVs dynamically and extends the N=308 base
from script 15 with the new labels. Produces a running comparison table:
  N=251 baseline (cluster-aware) → N=308 → N=3xx
(N=251, not N=260 -- see BASELINES comment below for why that distinction matters)

Usage
-----
  python code/16_incremental_retrain_with_new_labels.py

It automatically picks up any *_pilot_results.csv or *_results.csv in
  data/newdb_v2/labeling_candidates/
that have a non-empty Expert_Class column, builds the full merged set,
and runs the same cluster-aware grid + LOGO evaluation as scripts 09 and 15.

Outputs (version-tagged with actual N)
-------
  data/model_outputs/hyperparameter_search_results_N{N}.csv
  data/model_outputs/hyperparameter_search_summary_N{N}.json
  data/model_outputs/binary_hyperparameter_search_results_N{N}.csv
  data/model_outputs/binary_hyperparameter_search_summary_N{N}.json
  data/model_outputs/easy_hyperparameter_search_results_N{N}.csv
  data/model_outputs/easy_hyperparameter_search_summary_N{N}.json
  models/national_pilot_model_N{N}.joblib
  models/national_binary_model_N{N}.joblib
  models/national_easy_model_N{N}.joblib
  data/model_outputs/running_comparison_table.csv
"""
import glob, json, os, time, warnings
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
LAB_DIR  = os.path.join(BASE, "data", "newdb_v2", "labeling_candidates")
MOUT = os.path.join(BASE, "data", "model_outputs")
MMOD = os.path.join(BASE, "models")

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

# Running baselines (cluster-aware LOGO CV). NOTE: these are N=251 baselines from
# script 09 (mislabeled "N=260" in an earlier version -- corrected 2026-08-12). The
# "old" data below (combined_expert_labels.csv, 260 rows) includes 9 post-report BMK
# sites script 09 excluded, so any comparison against these numbers is confounded by
# that reintroduction, not a clean measure of the new labels alone.
BASELINES = {
    "3class":    {"N=251": 0.5776892430278885},
    "difficult": {"N=251": 0.7928286852589641},
    "easy":      {"N=251": 0.8207171314741036},
}

# ── 1. Old labels ──────────────────────────────────────────────────────────
print("=== 1. Old labels + features ===", flush=True)
old_labels = pd.read_csv(os.path.join(BASE, "data", "final", "combined_expert_labels.csv"))
old_feats  = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
old = old_labels.merge(
    old_feats[["Locality_ID","Latitude_WGS84","Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
print(f"  Old: {len(old)}", flush=True)

# ── 2. All new pilot results (auto-discovered) ─────────────────────────────
print("\n=== 2. Discovering new pilot result CSVs ===", flush=True)
result_files = glob.glob(os.path.join(LAB_DIR, "*results*.csv"))
new_frames = []
for fpath in sorted(result_files):
    df = pd.read_csv(fpath)
    if "Expert_Class" not in df.columns:
        continue
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    if len(labeled) == 0:
        continue
    src_tag = os.path.basename(fpath).replace(".csv","")
    labeled["Source"] = src_tag
    new_frames.append(labeled[["Locality_ID","Expert_Class","Source"]])
    print(f"  {src_tag}: {len(labeled)} labels", flush=True)

if not new_frames:
    print("  No new pilot results found — nothing to add. Run after filling batches.", flush=True)
    exit(0)

new_labels_raw = pd.concat(new_frames, ignore_index=True).drop_duplicates("Locality_ID")
# Source features from the authoritative main catalog, not the stale
# geosites_new_localities_features.csv intermediate (confirmed missing rows -- see 15).
new_feats = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
new = new_labels_raw.merge(
    new_feats[["Locality_ID","Latitude_WGS84","Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
# Remove any IDs already in old
new = new[~new["Locality_ID"].isin(set(old["Locality_ID"]))]
n_labeled_raw = len(new_labels_raw)
if len(new) != n_labeled_raw - len(set(new_labels_raw["Locality_ID"]) & set(old["Locality_ID"])):
    missing = set(new_labels_raw["Locality_ID"]) - set(new["Locality_ID"]) - set(old["Locality_ID"])
    if missing:
        print(f"  WARNING: {len(missing)} labeled row(s) dropped (no matching features): {missing}", flush=True)
print(f"  Net new labeled+featured: {len(new)}", flush=True)
print(f"  Classes: {new['Expert_Class'].value_counts().to_dict()}", flush=True)

# ── 3. Merge ───────────────────────────────────────────────────────────────
merged = pd.concat([
    old[["Locality_ID","Latitude_WGS84","Longitude_WGS84","Expert_Class"] + FEATURES],
    new[["Locality_ID","Latitude_WGS84","Longitude_WGS84","Expert_Class"] + FEATURES],
], ignore_index=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult","Difficult")
N = len(merged)
print(f"\n  Merged N={N}: {merged['Expert_Merged'].value_counts().to_dict()}", flush=True)

# ── 4. Cluster assignment ──────────────────────────────────────────────────
print("\n=== 3. Cluster assignment ===", flush=True)
lat, lon = merged["Latitude_WGS84"].values, merged["Longitude_WGS84"].values
n = len(merged)
def hav(lat, lon):
    R=6371000; lr=np.radians(lat); lo=np.radians(lon)
    dlat=lr[:,None]-lr[None,:]; dlon=lo[:,None]-lo[None,:]
    a=np.sin(dlat/2)**2+np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = hav(lat, lon)
parent = list(range(n))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(x,y):
    rx,ry=find(x),find(y)
    if rx!=ry: parent[rx]=ry
for i in range(n):
    for j in range(i+1,n):
        if D[i,j]<=500: union(i,j)
cluster_ids = np.array([find(i) for i in range(n)])
n_clusters = len(np.unique(cluster_ids))
print(f"  N={n}, clusters={n_clusters}", flush=True)

X = merged[FEATURES].values
logo = LeaveOneGroupOut()

# ── 5. Grid + eval helpers ─────────────────────────────────────────────────
def cv_score(factory, X, y, groups, n_repeats=5, n_splits=10):
    scores=[]
    for rep in range(n_repeats):
        skf=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr,te in skf.split(X,y,groups=groups):
            m=factory(); m.fit(X[tr],y[tr])
            scores.append(accuracy_score(y[te],m.predict(X[te])))
    return np.mean(scores), np.std(scores)

def run_grid(y, tag):
    results=[]
    for ne in [100,200,400]:
        for md in [3,4,5,6,7,8,None]:
            for msl in [1,2,3,5]:
                f=lambda ne=ne,md=md,msl=msl: RandomForestClassifier(
                    n_estimators=ne,max_depth=md,min_samples_leaf=msl,
                    class_weight="balanced",random_state=42,n_jobs=1)
                m,s=cv_score(f,X,y,cluster_ids)
                results.append({"model":"RF","n_est":ne,"max_depth":md,"min_leaf":msl,"mean_acc":m,"std_acc":s})
    print(f"[{tag}] RF [{time.time()-t0:.0f}s]", flush=True)
    for ne in [100,150,250]:
        for md in [3,4,5,6]:
            for lr in [0.03,0.08,0.15]:
                f=lambda ne=ne,md=md,lr=lr: XGBClassifier(
                    n_estimators=ne,max_depth=md,learning_rate=lr,
                    random_state=42,n_jobs=1,
                    eval_metric="mlogloss" if tag=="3class" else "logloss")
                m,s=cv_score(f,X,y,cluster_ids)
                results.append({"model":"XGB","n_est":ne,"max_depth":md,"lr":lr,"mean_acc":m,"std_acc":s})
    print(f"[{tag}] XGB [{time.time()-t0:.0f}s]", flush=True)
    for ne in [100,200]:
        for md in [2,3,4]:
            for lr in [0.05,0.1]:
                f=lambda ne=ne,md=md,lr=lr: GradientBoostingClassifier(
                    n_estimators=ne,max_depth=md,learning_rate=lr,random_state=42)
                m,s=cv_score(f,X,y,cluster_ids)
                results.append({"model":"GBM","n_est":ne,"max_depth":md,"lr":lr,"mean_acc":m,"std_acc":s})
    print(f"[{tag}] GBM [{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(results).sort_values("mean_acc",ascending=False)

def build_and_eval(df_res, y, tag, class_names):
    top_rf=df_res[df_res["model"]=="RF"].iloc[0]
    top_xgb=df_res[df_res["model"]=="XGB"].iloc[0]
    top_gbm=df_res[df_res["model"]=="GBM"].iloc[0]
    rf_f=RandomForestClassifier(
        n_estimators=int(top_rf["n_est"]),
        max_depth=None if pd.isna(top_rf["max_depth"]) else int(top_rf["max_depth"]),
        min_samples_leaf=int(top_rf["min_leaf"]),
        class_weight="balanced",random_state=42,n_jobs=1)
    xgb_f=XGBClassifier(
        n_estimators=int(top_xgb["n_est"]),max_depth=int(top_xgb["max_depth"]),
        learning_rate=top_xgb["lr"],random_state=42,n_jobs=1,
        eval_metric="mlogloss" if tag=="3class" else "logloss")
    gbm_f=GradientBoostingClassifier(
        n_estimators=int(top_gbm["n_est"]),max_depth=int(top_gbm["max_depth"]),
        learning_rate=top_gbm["lr"],random_state=42)
    preds=np.zeros(len(y))
    for tr,te in logo.split(X,y,groups=cluster_ids):
        m=VotingClassifier(estimators=[("rf",rf_f),("xgb",xgb_f),("gbm",gbm_f)],
                            voting="soft",n_jobs=1)
        m.fit(X[tr],y[tr]); preds[te]=m.predict(X[te])
    acc=accuracy_score(y,preds)
    print(f"[{tag}] Ensemble LOGO CV: {acc:.4f} ({int(acc*len(y))}/{len(y)})  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(confusion_matrix(y,preds), flush=True)
    print(classification_report(y,preds,target_names=class_names), flush=True)
    ens=VotingClassifier(estimators=[("rf",rf_f),("xgb",xgb_f),("gbm",gbm_f)],
                          voting="soft",n_jobs=1)
    ens.fit(X,y)
    return ens, acc

# ── 6. Run ─────────────────────────────────────────────────────────────────
print("\n=== 3-class ===", flush=True)
y3 = merged["Expert_Merged"].map({"Easy":0,"Moderate":1,"Difficult":2}).values
df3 = run_grid(y3,"3class")
df3.to_csv(os.path.join(MOUT,f"hyperparameter_search_results_N{N}.csv"),index=False)
ens3,acc3 = build_and_eval(df3,y3,"3class",["Easy","Moderate","Difficult"])
joblib.dump(ens3,os.path.join(MMOD,f"national_pilot_model_N{N}.joblib"))
json.dump({"N":N,"cluster_holdout_acc":acc3},
          open(os.path.join(MOUT,f"hyperparameter_search_summary_N{N}.json"),"w"),indent=2)

print("\n=== Binary Difficult ===", flush=True)
yd=(merged["Expert_Merged"]=="Difficult").astype(int).values
dfd=run_grid(yd,"difficult")
dfd.to_csv(os.path.join(MOUT,f"binary_hyperparameter_search_results_N{N}.csv"),index=False)
ensd,accd=build_and_eval(dfd,yd,"difficult",["Not-Difficult","Difficult"])
joblib.dump(ensd,os.path.join(MMOD,f"national_binary_model_N{N}.joblib"))
json.dump({"N":N,"cluster_holdout_acc":accd},
          open(os.path.join(MOUT,f"binary_hyperparameter_search_summary_N{N}.json"),"w"),indent=2)

print("\n=== Binary Easy ===", flush=True)
ye=(merged["Expert_Merged"]=="Easy").astype(int).values
dfe=run_grid(ye,"easy")
dfe.to_csv(os.path.join(MOUT,f"easy_hyperparameter_search_results_N{N}.csv"),index=False)
ense,acce=build_and_eval(dfe,ye,"easy",["Not-Easy","Easy"])
joblib.dump(ense,os.path.join(MMOD,f"national_easy_model_N{N}.joblib"))
json.dump({"N":N,"cluster_holdout_acc":acce},
          open(os.path.join(MOUT,f"easy_hyperparameter_search_summary_N{N}.json"),"w"),indent=2)

# ── 7. Running comparison table ────────────────────────────────────────────
print("\n"+"="*70, flush=True)
print(f"  RUNNING COMPARISON  (cluster-aware LOGO CV)", flush=True)
print("="*70, flush=True)
hdr = f"{'Target':<22}{'N=251':>10}{'N=308':>10}{'N='+str(N):>10}{'vs N=251':>10}"
print(hdr, flush=True)
print("-"*70, flush=True)
# Load N=308 results if available
def load_acc(summary_path):
    try:
        return json.load(open(summary_path)).get("cluster_holdout_acc")
    except Exception:
        return None

n308_3  = load_acc(os.path.join(MOUT,"hyperparameter_search_summary_N308.json"))
n308_d  = load_acc(os.path.join(MOUT,"binary_hyperparameter_search_summary_N308.json"))
n308_e  = load_acc(os.path.join(MOUT,"easy_hyperparameter_search_summary_N308.json"))

for lbl,base,n308,cur in [
    ("3-class",0.5776892430278885,n308_3,acc3),
    ("Binary Difficult",0.7928286852589641,n308_d,accd),
    ("Binary Easy",0.8207171314741036,n308_e,acce),
]:
    n308_str = f"{n308*100:.1f}%" if n308 else "  N/A "
    print(f"{lbl:<22}{base*100:>9.1f}%{n308_str:>10}{cur*100:>9.1f}%"
          f"{(cur-base)*100:>+9.1f}pp", flush=True)
print("="*70, flush=True)
print(f"Total time: {time.time()-t0:.0f}s  |  N={N}  |  clusters={n_clusters}", flush=True)
print("DONE.", flush=True)
