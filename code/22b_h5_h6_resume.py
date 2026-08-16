"""
Resume script (2026-08-14) -- only runs H5 (logistic regression baseline)
and H6 (Gaussian Process), the two pieces that never ran because
code/22_comprehensive_final.py crashed on H5's `multi_class` param (removed
in newer sklearn versions -- fixed here by just not passing it, sklearn
auto-selects multinomial for LogisticRegression when there are >2 classes).

Does NOT re-run H0-H4 -- those already completed successfully on your last
run (H0 500m baseline: 0.5771, confirmed from your pasted log). Re-derives
only what H5/H6 need (data load, features, 500m clustering) fresh, since
those are cheap (seconds), then runs just the two missing experiments.

Run (Z8, foreground fine):
    cd geosite_project1
    python .\\code\\22b_h5_h6_resume.py 2>&1 | Tee-Object -FilePath h5_h6_resume.log
Expect a few minutes for H5 (cheap), longer for H6 (GP, 10-fold group CV,
not full LOGO -- same reasoning as before, GP is O(n^3)/fold). If H6 also
throws a version-related error, paste it back before assuming anything else
is broken -- GaussianProcessClassifier's `multi_class` param is NOT the
same deprecated one LogisticRegression had, it should be fine, but flagging
the risk honestly rather than assuming.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
MOUT = os.path.join(BASE, "data", "model_outputs")
os.makedirs(MOUT, exist_ok=True)

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}
H0_500M_BASELINE = 0.5771  # from your completed code/22 run, tree ensemble, 500m LOGO-cluster CV

log("Loading labels")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    if "Confidence" not in labeled.columns:
        labeled["Confidence"] = "Medium"
    frames.append(labeled[["Locality_ID", "Expert_Class", "Confidence"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["conf_weight"] = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7)
N = len(merged)
log(f"N={N}")

y3 = merged["Expert_Merged"].map({"Easy": 0, "Moderate": 1, "Difficult": 2}).values
conf_w = merged["conf_weight"].values

def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cluster_at_radius(D, radius_m):
    n = D.shape[0]
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

FULL_D = haversine_matrix(merged["Latitude_WGS84"].values, merged["Longitude_WGS84"].values)
cluster_500 = cluster_at_radius(FULL_D, 500)
X_scaled = StandardScaler().fit_transform(merged[FEATURES].values)

def combined_weight(y_tr, conf_tr):
    return conf_tr * compute_sample_weight("balanced", y_tr)

ALL = {}

# ── H5: logistic regression baseline (fixed: no `multi_class` kwarg) ──────────
print("\n"+"="*70, flush=True); print("H5 -- LOGISTIC REGRESSION BASELINE (fixed)", flush=True); print("="*70, flush=True)
preds_lr = np.zeros(N, dtype=int)
for tr, te in LeaveOneGroupOut().split(X_scaled, y3, groups=cluster_500):
    sw = combined_weight(y3[tr], conf_w[tr])
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_scaled[tr], y3[tr], sample_weight=sw)
    preds_lr[te] = lr.predict(X_scaled[te])
acc_lr = accuracy_score(y3, preds_lr)
log(f"  Logistic regression LOGO-cluster CV acc={acc_lr:.4f} (tree ensemble H0 500m was {H0_500M_BASELINE})")
ALL["H5_logistic_regression_baseline"] = dict(acc_logo_cluster=round(acc_lr,4))

# ── H6: Gaussian Process, 10-fold group CV (not full LOGO, GP is O(n^3)/fold) ──
print("\n"+"="*70, flush=True); print("H6 -- GAUSSIAN PROCESS (RBF kernel, 10-fold group CV)", flush=True); print("="*70, flush=True)
preds_gp = np.full(N, -1, dtype=int)
kernel = 1.0 * RBF(length_scale=1.0)
t_gp = time.time()
gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
n_gp_folds = 0
for tr, te in gkf.split(X_scaled, y3, groups=cluster_500):
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1, multi_class="one_vs_rest")
    gp.fit(X_scaled[tr], y3[tr])
    preds_gp[te] = gp.predict(X_scaled[te])
    n_gp_folds += 1
    log(f"    GP fold {n_gp_folds}/10 done  [{time.time()-t_gp:.0f}s]")
covered = preds_gp >= 0
acc_gp = accuracy_score(y3[covered], preds_gp[covered])
log(f"  Gaussian Process 10-fold group CV acc={acc_gp:.4f} (covered {covered.sum()}/{N})  [{time.time()-t_gp:.0f}s]  "
    f"(tree ensemble H0 500m was {H0_500M_BASELINE})")
ALL["H6_gaussian_process"] = dict(acc_group_cv_10fold=round(acc_gp,4), n_covered=int(covered.sum()))

out_path = os.path.join(MOUT, f"h5_h6_resume_results_N{N}.json")
json.dump(ALL, open(out_path, "w"), indent=2, default=str)
print(f"\nSaved to {out_path}. Total elapsed: {time.time()-t0:.0f}s", flush=True)
