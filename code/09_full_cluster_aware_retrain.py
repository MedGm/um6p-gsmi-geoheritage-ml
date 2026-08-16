"""
Full, methodologically-consistent retrain: hyperparameter tuning AND final evaluation
both respect locality clusters (sites <=500m apart grouped together), closing the
leakage gap end-to-end rather than patching only the final evaluation. Adds the newly
recovered Dist_to_Settlement_m as a 6th feature. Runs on the official N=251 report
dataset (9 post-report BMK additions excluded) for all three targets: 3-class,
binary-Difficult, binary-Easy.

n_jobs=1 throughout: at N=251 a single tree-model fit is milliseconds, and the grid
does ~20k fits total -- spawning a process pool per fit (n_jobs=-1) makes pool-spawn
overhead dominate over actual compute, which is what stalled this for 4h on the Z8
without finishing. Single-threaded fits are the fast path at this scale.
"""
import pandas as pd, numpy as np, joblib, json, time, glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

t0 = time.time()

POST_REPORT_ADDITIONS = {'loc_00233','loc_00229','loc_00231','loc_00236','loc_00227',
                          'loc_00228','loc_00291','loc_00230','loc_00296'}
full = pd.read_csv('data/final/geosites_mcdm_national.csv')
ttah = pd.read_csv('data/final/regional_label_sources/ttah_expert_labels.csv')[['Locality_ID','Expert_Class']]
others = []
for f in glob.glob('data/final/regional_label_sources/*.csv'):
    if 'ttah' in f: continue
    d = pd.read_csv(f)
    others.append(d[['Locality_ID','Expert_Class']].dropna(subset=['Expert_Class']))
combined = pd.concat([ttah] + others, ignore_index=True).drop_duplicates(subset=['Locality_ID'])
combined = combined[~combined['Locality_ID'].isin(POST_REPORT_ADDITIONS)]
assert len(combined) == 251, f"Expected 251, got {len(combined)}"

FEATURES = ['Dist_to_Highway_m','Slope_deg','Ruggedness','Elevation_m','LULC_Friction','Dist_to_Settlement_m']
labeled = combined.merge(full, on='Locality_ID')
labeled['Expert_Merged'] = labeled['Expert_Class'].replace('Very Difficult','Difficult')
labeled = labeled.reset_index(drop=True)
print(f"N={len(labeled)}, features={FEATURES}", flush=True)

lat, lon = labeled['Latitude_WGS84'].values, labeled['Longitude_WGS84'].values
n = len(labeled)
def haversine_matrix(lat, lon):
    R = 6371000
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    dlat = lat_r[:,None]-lat_r[None,:]; dlon = lon_r[:,None]-lon_r[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:,None])*np.cos(lat_r[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = haversine_matrix(lat, lon)
parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry
for i in range(n):
    for j in range(i+1, n):
        if D[i,j] <= 500: union(i, j)
cluster_ids = np.array([find(i) for i in range(n)])
n_clusters = len(np.unique(cluster_ids))
print(f"Locality clusters: {n_clusters}", flush=True)

X = labeled[FEATURES].values
logo = LeaveOneGroupOut()

def cluster_aware_cv_score(model, X, y, groups, n_repeats=5, n_splits=10):
    scores = []
    for rep in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, y, groups=groups):
            m = model()
            m.fit(X[tr], y[tr])
            scores.append(accuracy_score(y[te], m.predict(X[te])))
    return np.mean(scores), np.std(scores)

def run_grid(y, tag):
    results = []
    for n_est in [100, 200, 400]:
        for md in [3, 4, 5, 6, 7, 8, None]:
            for msl in [1, 2, 3, 5]:
                factory = lambda n_est=n_est, md=md, msl=msl: RandomForestClassifier(
                    n_estimators=n_est, max_depth=md, min_samples_leaf=msl,
                    class_weight='balanced', random_state=42, n_jobs=1)
                mean_acc, std_acc = cluster_aware_cv_score(factory, X, y, cluster_ids)
                results.append({'model':'RF','n_est':n_est,'max_depth':md,'min_leaf':msl,'mean_acc':mean_acc,'std_acc':std_acc})
    print(f"[{tag}] RF grid done [{time.time()-t0:.0f}s]", flush=True)
    for n_est in [100, 150, 250]:
        for md in [3, 4, 5, 6]:
            for lr in [0.03, 0.08, 0.15]:
                factory = lambda n_est=n_est, md=md, lr=lr: XGBClassifier(
                    n_estimators=n_est, max_depth=md, learning_rate=lr, random_state=42,
                    eval_metric='mlogloss' if tag=='3class' else 'logloss', n_jobs=1)
                mean_acc, std_acc = cluster_aware_cv_score(factory, X, y, cluster_ids)
                results.append({'model':'XGB','n_est':n_est,'max_depth':md,'lr':lr,'mean_acc':mean_acc,'std_acc':std_acc})
    print(f"[{tag}] XGB grid done [{time.time()-t0:.0f}s]", flush=True)
    for n_est in [100, 200]:
        for md in [2, 3, 4]:
            for lr in [0.05, 0.1]:
                factory = lambda n_est=n_est, md=md, lr=lr: GradientBoostingClassifier(
                    n_estimators=n_est, max_depth=md, learning_rate=lr, random_state=42)
                mean_acc, std_acc = cluster_aware_cv_score(factory, X, y, cluster_ids)
                results.append({'model':'GBM','n_est':n_est,'max_depth':md,'lr':lr,'mean_acc':mean_acc,'std_acc':std_acc})
    print(f"[{tag}] GBM grid done [{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(results).sort_values('mean_acc', ascending=False)

def build_and_eval(df_results, y, tag, class_names):
    top_rf = df_results[df_results['model']=='RF'].iloc[0]
    top_xgb = df_results[df_results['model']=='XGB'].iloc[0]
    top_gbm = df_results[df_results['model']=='GBM'].iloc[0]
    rf_f = RandomForestClassifier(n_estimators=int(top_rf['n_est']), max_depth=None if pd.isna(top_rf['max_depth']) else int(top_rf['max_depth']),
                                   min_samples_leaf=int(top_rf['min_leaf']), class_weight='balanced', random_state=42, n_jobs=1)
    xgb_f = XGBClassifier(n_estimators=int(top_xgb['n_est']), max_depth=int(top_xgb['max_depth']), learning_rate=top_xgb['lr'],
                           random_state=42, eval_metric='mlogloss' if tag=='3class' else 'logloss', n_jobs=1)
    gbm_f = GradientBoostingClassifier(n_estimators=int(top_gbm['n_est']), max_depth=int(top_gbm['max_depth']), learning_rate=top_gbm['lr'], random_state=42)
    preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups=cluster_ids):
        m = VotingClassifier(estimators=[('rf',rf_f),('xgb',xgb_f),('gbm',gbm_f)], voting='soft', n_jobs=1)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    acc = accuracy_score(y, preds)
    print(f"[{tag}] Tuned ensemble CLUSTER-HOLDOUT CV: {acc:.4f} ({int(acc*len(y))}/{len(y)})  [{time.time()-t0:.0f}s]", flush=True)
    print(confusion_matrix(y, preds), flush=True)
    print(classification_report(y, preds, target_names=class_names), flush=True)
    ens_full = VotingClassifier(estimators=[('rf',rf_f),('xgb',xgb_f),('gbm',gbm_f)], voting='soft', n_jobs=1)
    ens_full.fit(X, y)
    return ens_full, acc, {'best_rf':top_rf.to_dict(),'best_xgb':top_xgb.to_dict(),'best_gbm':top_gbm.to_dict(),'cluster_holdout_acc':acc}

y3 = labeled['Expert_Merged'].map({'Easy':0,'Moderate':1,'Difficult':2}).values
df3 = run_grid(y3, '3class')
df3.to_csv('data/model_outputs/hyperparameter_search_results_clusteraware.csv', index=False)
ens3, acc3, summ3 = build_and_eval(df3, y3, '3class', ['Easy','Moderate','Difficult'])
joblib.dump(ens3, 'models/national_pilot_model_clusteraware.joblib')
json.dump(summ3, open('data/model_outputs/hyperparameter_search_summary_clusteraware.json','w'), indent=2, default=str)

yd = (labeled['Expert_Merged']=='Difficult').astype(int).values
dfd = run_grid(yd, 'difficult')
dfd.to_csv('data/model_outputs/binary_hyperparameter_search_results_clusteraware.csv', index=False)
ensd, accd, summd = build_and_eval(dfd, yd, 'difficult', ['Not-Difficult','Difficult'])
joblib.dump(ensd, 'models/national_binary_model_clusteraware.joblib')
json.dump(summd, open('data/model_outputs/binary_hyperparameter_search_summary_clusteraware.json','w'), indent=2, default=str)

ye = (labeled['Expert_Merged']=='Easy').astype(int).values
dfe = run_grid(ye, 'easy')
dfe.to_csv('data/model_outputs/easy_hyperparameter_search_results_clusteraware.csv', index=False)
ense, acce, summe = build_and_eval(dfe, ye, 'easy', ['Not-Easy','Easy'])
joblib.dump(ense, 'models/national_easy_model_clusteraware.joblib')
json.dump(summe, open('data/model_outputs/easy_hyperparameter_search_summary_clusteraware.json','w'), indent=2, default=str)

print("\n=== FINAL SUMMARY (cluster-aware tuning + cluster-holdout evaluation, 6 features incl. settlement) ===", flush=True)
print(f"3-class:            {acc3:.4f}", flush=True)
print(f"Binary Difficult:    {accd:.4f}", flush=True)
print(f"Binary Easy:         {acce:.4f}", flush=True)
print(f"Total time: {time.time()-t0:.0f}s", flush=True)
print("DONE.", flush=True)
