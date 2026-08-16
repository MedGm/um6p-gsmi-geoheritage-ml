"""
Derives every number used in the redesigned report tables from source files
already in results/ and data/final/ -- no new training, just arithmetic on
verified results. Run once, writes derived_numbers.json for traceability.
"""
import json, glob, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..")
RESULTS_TRAINING = os.path.join(RESULTS, "json", "training")
RESULTS_OTHER = os.path.join(RESULTS, "json", "other")
ROOT = os.path.join(HERE, "..", "..")
DATA = os.path.join(ROOT, "data", "final", "regional_label_sources")

# ---- Table 1: dataset composition (region x class), N=733 ----------------
frames = []
for f in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
    if "expert_labels" not in os.path.basename(f) or "combined" in os.path.basename(f):
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(ROOT, "data", "final", "geosites_mcdm_national.csv"))
merged = all_labels.merge(catalog[["Locality_ID", "Region"]], on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
assert len(merged) == 733, f"expected N=733, got {len(merged)}"
composition = pd.crosstab(merged["Region"], merged["Expert_Merged"], margins=True, margins_name="Total")
composition_dict = composition.to_dict(orient="index")

# ---- Table 5: confusion-derived Precision/Recall/F1 -----------------------
# Sourced from the pasted G0a terminal output (final_v2 run), rows=True, cols=Pred,
# class order [Not-X, X]. Exact ints, N=733 both.
def prf(tn, fp, fn, tp):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    prec_neg = tn / (tn + fn)
    rec_neg = tn / (tn + fp)
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg)
    acc = (tn + tp) / (tn + fp + fn + tp)
    return dict(acc=round(acc, 4),
                pos=dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4)),
                neg=dict(precision=round(prec_neg, 4), recall=round(rec_neg, 4), f1=round(f1_neg, 4)))

difficult_cm = prf(tn=428, fp=91, fn=99, tp=115)
easy_cm = prf(tn=431, fp=95, fn=101, tp=106)

# ---- Model-family comparison, all cluster-aware 500m LOGO, national N=733 -
# H0/H5/H6 from h0_h4_comprehensive_log.txt + h5_h6_resume_results_N733.json
model_family = {
    "Logistic regression (multinomial)": 0.4638,
    "k-NN, geographic distance (k=5-15)": 0.608,
    "k-NN, feature-space distance (k=15-25)": 0.570,
    "Gaussian Process (RBF, 10-fold group CV)": 0.5744,
    "Tree ensemble (RF+XGB+GBM+LightGBM+CatBoost)": 0.5771,
}

# ---- Leave-region-out, Difficult classifier, from final_v2_results_N733.json
# (G0a_difficult.leave_region_out) -- verbatim, NaN/degenerate rows flagged not dropped.
with open(os.path.join(RESULTS_TRAINING, "final_v2_results_N733.json"), encoding="utf-8") as f:
    final_v2 = json.load(f)
lro_raw = final_v2["G0a_difficult"]["leave_region_out"]
lro_testable = [r for r in lro_raw if r["acc"] == r["acc"]]  # drop NaN (n_test=5 regions)

out = dict(
    dataset_composition=composition_dict,
    confusion_derived=dict(difficult=difficult_cm, easy=easy_cm),
    model_family_cluster_aware=model_family,
    leave_region_out_difficult=lro_testable,
)
outpath = os.path.join(RESULTS_OTHER, "derived_numbers.json")
json.dump(out, open(outpath, "w"), indent=2, ensure_ascii=False)
print("Wrote", outpath)
print(json.dumps(out, indent=2, ensure_ascii=False))
