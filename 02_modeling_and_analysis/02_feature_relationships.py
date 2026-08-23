"""
data_audit/02_feature_relationships.py  (2026-08-18)

Phase 3: feature relationships, on the full 939-site labeled set (733 original
+ 206 el_ouali_2026 batch). Checks:
  A. Correlation / multicollinearity among the 6 production features (Pearson,
     Spearman, and VIF -- VIF catches 3+-way redundancy that pairwise
     correlation can miss)
  B. Feature-vs-class relationship strength: one-way ANOVA F-stat and mutual
     information, per feature
  C. Geosite_Type vs class (chi-square, ~33% coverage) -- does the categorical
     type info (hydric/karst/etc, motivated by the Phase-2 finding that these
     types systematically run Difficult regardless of terrain features) carry
     independent signal
  D. Geological_Domain vs class (chi-square, ~97% coverage -- much better
     usable coverage than Geosite_Type)

No model is trained here -- this is purely descriptive, to decide what (if
anything) is worth engineering into Phase 5.
"""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = HERE

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
# Geosite_Type/Geological_Domain live in the master Excel (joined from the locality
# tables in code/35), not in the production catalog CSV itself.
master = pd.read_excel(os.path.join(BASE, "geosites_master_1667_with_accessibility.xlsx"))
type_domain = master[["Locality_ID", "Geosite_Type", "Geological_Domain"]]

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
    catalog[["Locality_ID", "Region", "Geosite_Name"] + FEATURES],
    on="Locality_ID", how="inner").merge(type_domain, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
print(f"N={N}\n")

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

# ============================================================ A. Correlation + VIF
log("=" * 70)
log("A. FEATURE CORRELATION + MULTICOLLINEARITY (VIF)")
log("=" * 70)
X = merged[FEATURES]
log("\nPearson correlation:")
log(X.corr(method="pearson").round(3).to_string())
log("\nSpearman correlation (rank-based, catches non-linear monotonic relationships):")
log(X.corr(method="spearman").round(3).to_string())

from statsmodels.stats.outliers_influence import variance_inflation_factor
Xs = (X - X.mean()) / X.std()
Xs_const = Xs.copy()
Xs_const.insert(0, "const", 1.0)
log("\nVariance Inflation Factor (VIF) per feature -- >5 is commonly flagged as concerning, >10 as serious:")
for i, feat in enumerate(FEATURES, start=1):
    vif = variance_inflation_factor(Xs_const.values, i)
    flag = "  <-- CONCERNING" if vif > 5 else ""
    log(f"  {feat:22s} VIF={vif:.2f}{flag}")

# ============================================================ B. Feature vs class strength
log("\n" + "=" * 70)
log("B. FEATURE-VS-CLASS RELATIONSHIP STRENGTH")
log("=" * 70)
y = merged["Expert_Merged"].values
le = LabelEncoder()
y_enc = le.fit_transform(y)

log(f"\n{'Feature':22s} {'ANOVA F':>10s} {'ANOVA p':>10s} {'Mutual Info':>12s}")
mi = mutual_info_classif(X.values, y_enc, random_state=42)
for i, feat in enumerate(FEATURES):
    groups = [merged.loc[merged['Expert_Merged'] == c, feat].values for c in ['Easy', 'Moderate', 'Difficult']]
    f_stat, p_val = stats.f_oneway(*groups)
    log(f"{feat:22s} {f_stat:10.2f} {p_val:10.4f} {mi[i]:12.4f}")
log("\n(ANOVA F/p: is the feature's MEAN different across the 3 classes -- high F, low p = strong linear")
log("separation. Mutual Info: any statistical dependency, linear or not -- 0 = independent, higher = more")
log("informative. Both computed the same way code/19-27's model-family comparisons implicitly rely on.)")

# ============================================================ C. Geosite_Type vs class
log("\n" + "=" * 70)
log("C. GEOSITE_TYPE vs ACCESSIBILITY CLASS (chi-square, coverage-limited)")
log("=" * 70)
has_type = merged["Geosite_Type"].notna()
log(f"Coverage: {has_type.sum()}/{N} ({100*has_type.sum()/N:.1f}%)")
if has_type.sum() > 30:
    ct = pd.crosstab(merged.loc[has_type, "Geosite_Type"], merged.loc[has_type, "Expert_Merged"])
    # keep only types with enough sites to be informative
    ct = ct[ct.sum(axis=1) >= 5]
    log(f"\nCross-tab (types with >=5 labeled sites):\n{ct.to_string()}")
    if ct.shape[0] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        log(f"\nChi-square: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
        log("(p<0.05 -> Geosite_Type and accessibility class are NOT independent -- type carries signal)")

# ============================================================ D. Geological_Domain vs class
log("\n" + "=" * 70)
log("D. GEOLOGICAL_DOMAIN vs ACCESSIBILITY CLASS (chi-square, near-full coverage)")
log("=" * 70)
has_domain = merged["Geological_Domain"].notna()
log(f"Coverage: {has_domain.sum()}/{N} ({100*has_domain.sum()/N:.1f}%)")
ct2 = pd.crosstab(merged.loc[has_domain, "Geological_Domain"], merged.loc[has_domain, "Expert_Merged"])
ct2 = ct2[ct2.sum(axis=1) >= 5]
log(f"\nCross-tab (domains with >=5 labeled sites):\n{ct2.to_string()}")
if ct2.shape[0] > 1:
    chi2, p, dof, _ = stats.chi2_contingency(ct2)
    log(f"\nChi-square: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
    log("(p<0.05 -> Geological_Domain and accessibility class are NOT independent -- domain carries signal)")
    # per-domain Difficult rate, sorted, for interpretability
    rate = ct2.div(ct2.sum(axis=1), axis=0)
    if "Difficult" in rate.columns:
        log("\nDifficult-rate by domain (sorted):")
        log(rate["Difficult"].sort_values(ascending=False).round(3).to_string())

with open(os.path.join(OUT, "phase3_feature_relationships_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log("\nDONE -- Phase 3 descriptive analysis complete. No model trained, no data modified.")
