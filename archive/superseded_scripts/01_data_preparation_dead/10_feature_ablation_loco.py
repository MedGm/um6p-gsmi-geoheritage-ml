"""
Wrapper-based feature ablation study: Leave-One-Covariate-Out (LOCO) + marginal/
univariate screening, scored under grouped cross-validation. Same method used on the
sibling Cr(VI) adsorption project.

Reuses code/09_full_cluster_aware_retrain.py's exact cluster construction (500m
single-linkage via haversine + union-find, official N=251 report dataset) and its
LeaveOneGroupOut cluster-holdout CV protocol -- no new validation protocol invented.

Design choice, disclosed up front: hyperparameters are held FIXED per target (pulled
from the already-completed cluster-aware tuning in
data/model_outputs/*_clusteraware.json) across every one of the 25 feature configs,
rather than re-tuned per config. Two reasons: (1) re-tuning per config would confound
"this feature helps" with "this particular hyperparameter search got lucky" -- holding
the model fixed isolates the feature's marginal contribution cleanly, which is the
actual point of a wrapper-method LOCO study; (2) a fully re-tuned 148-config grid search
per feature-set took ~2h for ONE feature set in code/09 -- re-tuning all 25 would take
days. This is a deliberate methodological choice, not an oversight, and is reported as
such.

Candidate columns tested (from data/final/geosites_mcdm_national.csv, never LOCO-tested
before): Dist_to_Dam_m, Dist_to_River_m, Soil_Class, Geology_Class, Coordinate_Precision,
N_Raster_Cells_Imputed. MCDM_Score/AHP_Score/*_Class/National_Model_Class are explicitly
EXCLUDED as candidates -- they are formula outputs derived from these same raw columns,
so using them as model inputs would be circular (the same circularity failure mode this
project's origin story already found and fixed once with the 91.9%-accuracy label bug).

Two-stage protocol per config x target:
  1. Primary score: single-pass LeaveOneGroupOut cluster-holdout CV (accuracy + macro-F1).
  2. Stability check (only for configs that beat-or-match the baseline's primary score):
     100 repeated random leave-N-clusters-out draws (20% of clusters held out each draw,
     fresh model fit each time) -- reports median, std, and 5th-percentile (worst-case),
     not just a single number, so a config that wins once on a lucky split isn't mistaken
     for a genuine improvement.
"""
import pandas as pd, numpy as np, glob, json, time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Reconstruct the official N=251 labeled dataset (identical to code/09)
# ---------------------------------------------------------------------------
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

labeled = combined.merge(full, on='Locality_ID').reset_index(drop=True)
labeled['Expert_Merged'] = labeled['Expert_Class'].replace('Very Difficult','Difficult')
print(f"N={len(labeled)}", flush=True)

# Candidate column prep: encode Coordinate_Precision (constant 'point' for all rows --
# will be reported as degenerate/zero-variance, not silently dropped), median-impute
# Geology_Class's 2 NaNs.
labeled['Coordinate_Precision'] = labeled['Coordinate_Precision'].map({'point': 0}).fillna(0)
labeled['Geology_Class'] = labeled['Geology_Class'].fillna(labeled['Geology_Class'].median())

BASELINE_FEATURES = ['Dist_to_Highway_m','Slope_deg','Ruggedness','Elevation_m','LULC_Friction','Dist_to_Settlement_m']
CANDIDATES = ['Dist_to_Dam_m','Dist_to_River_m','Soil_Class','Geology_Class','Coordinate_Precision','N_Raster_Cells_Imputed']

for c in CANDIDATES:
    assert labeled[c].isna().sum() == 0, f"{c} still has NaNs after prep"

# ---------------------------------------------------------------------------
# 2. 500m single-linkage locality clusters (identical construction to code/09)
# ---------------------------------------------------------------------------
def haversine_matrix(lat, lon):
    R = 6371000
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    dlat = lat_r[:,None]-lat_r[None,:]; dlon = lon_r[:,None]-lon_r[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:,None])*np.cos(lat_r[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

lat, lon = labeled['Latitude_WGS84'].values, labeled['Longitude_WGS84'].values
n = len(labeled)
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
labeled['cluster_id'] = [find(i) for i in range(n)]
groups = labeled['cluster_id'].values
n_clusters = len(np.unique(groups))
print(f"Locality clusters: {n_clusters}", flush=True)

logo = LeaveOneGroupOut()

# ---------------------------------------------------------------------------
# 3. Targets + fixed hyperparameters (from data/model_outputs/*_clusteraware.json,
#    the already-completed cluster-aware retrain -- held fixed across all configs)
# ---------------------------------------------------------------------------
TARGETS = {
    '3class': {
        'y': labeled['Expert_Merged'].map({'Easy':0,'Moderate':1,'Difficult':2}).values,
        'eval_metric': 'mlogloss',
        'rf': dict(n_estimators=200, max_depth=6, min_samples_leaf=1),
        'xgb': dict(n_estimators=100, max_depth=4, learning_rate=0.03),
        'gbm': dict(n_estimators=100, max_depth=4, learning_rate=0.05),
    },
    'difficult': {
        'y': (labeled['Expert_Merged']=='Difficult').astype(int).values,
        'eval_metric': 'logloss',
        'rf': dict(n_estimators=200, max_depth=7, min_samples_leaf=1),
        'xgb': dict(n_estimators=100, max_depth=4, learning_rate=0.03),
        'gbm': dict(n_estimators=100, max_depth=2, learning_rate=0.05),
    },
    'easy': {
        'y': (labeled['Expert_Merged']=='Easy').astype(int).values,
        'eval_metric': 'logloss',
        'rf': dict(n_estimators=200, max_depth=None, min_samples_leaf=1),
        'xgb': dict(n_estimators=150, max_depth=4, learning_rate=0.03),
        'gbm': dict(n_estimators=200, max_depth=2, learning_rate=0.1),
    },
}

def build_ensemble(target_key):
    spec = TARGETS[target_key]
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1, **spec['rf'])
    xgb = XGBClassifier(random_state=42, eval_metric=spec['eval_metric'], n_jobs=1, **spec['xgb'])
    gbm = GradientBoostingClassifier(random_state=42, **spec['gbm'])
    return VotingClassifier(estimators=[('rf',rf),('xgb',xgb),('gbm',gbm)], voting='soft', n_jobs=1)

def cluster_holdout_score(X, y, target_key):
    preds = np.zeros(len(y), dtype=int)
    for tr, te in logo.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            preds[te] = y[tr][0]  # degenerate fold guard, shouldn't trigger given class sizes
            continue
        m = build_ensemble(target_key)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    acc = accuracy_score(y, preds)
    f1m = f1_score(y, preds, average='macro', zero_division=0)
    return acc, f1m

def repeated_holdout_stability(X, y, target_key, n_repeats=100, holdout_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_hold = max(1, int(len(unique_groups) * holdout_frac))
    accs = []
    for _ in range(n_repeats):
        held = rng.choice(unique_groups, size=n_hold, replace=False)
        mask_te = np.isin(groups, held)
        mask_tr = ~mask_te
        if mask_te.sum() == 0 or mask_tr.sum() == 0 or len(np.unique(y[mask_tr])) < 2:
            continue
        m = build_ensemble(target_key)
        m.fit(X[mask_tr], y[mask_tr])
        pred = m.predict(X[mask_te])
        accs.append(accuracy_score(y[mask_te], pred))
    accs = np.array(accs)
    return {'median': float(np.median(accs)), 'std': float(np.std(accs)),
            'p5': float(np.percentile(accs, 5)), 'n_valid_draws': len(accs)}

# ---------------------------------------------------------------------------
# 4. Build the config sweep: baseline, LOCO (drop 1 of 6), LOCO-in (add 1 candidate
#    to the 6), marginal solo-1-feature, marginal solo-2-feature (candidate + the
#    dominant baseline feature Dist_to_Highway_m)
# ---------------------------------------------------------------------------
configs = [{'name': 'baseline_6feat', 'features': BASELINE_FEATURES.copy(), 'type': 'baseline'}]
for f in BASELINE_FEATURES:
    configs.append({'name': f'LOCO_drop_{f}', 'features': [x for x in BASELINE_FEATURES if x != f], 'type': 'LOCO'})
for c in CANDIDATES:
    configs.append({'name': f'LOCOin_add_{c}', 'features': BASELINE_FEATURES + [c], 'type': 'LOCO-in'})
for c in CANDIDATES:
    configs.append({'name': f'solo_{c}', 'features': [c], 'type': 'marginal-1feat'})
    configs.append({'name': f'solo_{c}_plus_highway', 'features': [c, 'Dist_to_Highway_m'], 'type': 'marginal-2feat'})

print(f"Total configs: {len(configs)}, targets: {len(TARGETS)}, primary evals: {len(configs)*len(TARGETS)}", flush=True)

# ---------------------------------------------------------------------------
# 5. Primary sweep
# ---------------------------------------------------------------------------
rows = []
baseline_acc = {}
for target_key, spec in TARGETS.items():
    y = spec['y']
    for cfg in configs:
        X = labeled[cfg['features']].values
        acc, f1m = cluster_holdout_score(X, y, target_key)
        rows.append({'target': target_key, 'config': cfg['name'], 'type': cfg['type'],
                     'n_features': len(cfg['features']), 'features': ';'.join(cfg['features']),
                     'cv_accuracy': acc, 'cv_macro_f1': f1m})
        if cfg['name'] == 'baseline_6feat':
            baseline_acc[target_key] = acc
        print(f"[{target_key}] {cfg['name']:35s} n_feat={len(cfg['features'])}  acc={acc:.4f}  macroF1={f1m:.4f}  [{time.time()-t0:.0f}s]", flush=True)

results = pd.DataFrame(rows)
results.to_csv('data/model_outputs/feature_ablation_loco_primary.csv', index=False)
print(f"\nPrimary sweep done. Baseline accuracy per target: {baseline_acc}  [{time.time()-t0:.0f}s]", flush=True)

# ---------------------------------------------------------------------------
# 6. Stability check for every config that beats-or-matches baseline (plus baseline
#    itself, as the reference distribution)
# ---------------------------------------------------------------------------
stability_rows = []
for target_key, spec in TARGETS.items():
    y = spec['y']
    qualifying = results[(results['target']==target_key) &
                          ((results['cv_accuracy'] >= baseline_acc[target_key]) |
                           (results['config']=='baseline_6feat'))]
    for _, r in qualifying.iterrows():
        feats = r['features'].split(';')
        X = labeled[feats].values
        stab = repeated_holdout_stability(X, y, target_key)
        stability_rows.append({'target': target_key, 'config': r['config'], 'type': r['type'],
                               'n_features': r['n_features'], 'primary_cv_accuracy': r['cv_accuracy'],
                               **stab})
        print(f"[stability][{target_key}] {r['config']:35s} median={stab['median']:.4f} "
              f"std={stab['std']:.4f} p5(worst-case)={stab['p5']:.4f} n_draws={stab['n_valid_draws']}  [{time.time()-t0:.0f}s]", flush=True)

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv('data/model_outputs/feature_ablation_loco_stability.csv', index=False)

# ---------------------------------------------------------------------------
# 7. Recommendation per target: an alt config is only called a genuine win if its
#    worst-realistic-case (median - std) still clears the baseline's median -- a
#    conservative bar, deliberately, to avoid calling a lucky split a real improvement.
# ---------------------------------------------------------------------------
print("\n=== RECOMMENDATION ===", flush=True)
for target_key in TARGETS:
    sub = stability_df[stability_df['target']==target_key]
    base_row = sub[sub['config']=='baseline_6feat'].iloc[0]
    challengers = sub[sub['config']!='baseline_6feat']
    winners = challengers[(challengers['median'] - challengers['std']) > base_row['median']]
    print(f"\n[{target_key}] baseline: median={base_row['median']:.4f} std={base_row['std']:.4f} p5={base_row['p5']:.4f}", flush=True)
    if len(winners) == 0:
        print(f"[{target_key}] RECOMMENDATION: keep current 6-feature baseline. "
              f"No challenger config's worst-realistic-case beats baseline's median under repeated holdout.", flush=True)
    else:
        for _, w in winners.sort_values('median', ascending=False).iterrows():
            print(f"[{target_key}] CANDIDATE WIN: {w['config']} median={w['median']:.4f} std={w['std']:.4f} "
                  f"p5={w['p5']:.4f} (n_draws={w['n_valid_draws']}) -- verify with more repeats/seeds before adopting.", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
print("DONE.", flush=True)
