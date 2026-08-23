"""
data_audit/22_calibration_conformal_n939.py  (2026-08-22)

Calibration/conformal analysis rerun on N=939 (was N=733 in
code/25_oof_calibration_conformal.py). No retraining needed -- reuses the
already-saved Baseline_939 per-site OOF probabilities
(results/json/other/phase5_{difficult,easy}_oof_per_site.csv, computed by
data_audit/07/08 under the identical LOGO-cluster CV protocol, no confidence
weighting -- current project policy, an improvement on the old script's
confidence-weighted OOF). Same brier_ece/isotonic/split-conformal methodology
as code/25, just applied to the current data.

Output:
  results/json/other/oof_probabilities_n939.csv
  results/json/other/calibration_conformal_results_n939.json
"""
import json, os, time
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(BASE, "results", "json", "other")

def brier_ece(y, p, n_bins=10):
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        w = mask.sum() / len(p)
        ece += w * abs(conf - acc)
        reliability.append({"bin": [round(lo,2), round(hi,2)], "n": int(mask.sum()),
                             "mean_predicted": round(float(conf), 4), "empirical_freq": round(float(acc), 4)})
    return brier, float(ece), reliability

def split_conformal(y, p, alpha=0.10, seed=42):
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    half = n // 2
    cal_idx, val_idx = idx[:half], idx[half:]
    p_true_cal = np.where(y[cal_idx] == 1, p[cal_idx], 1 - p[cal_idx])
    scores_cal = 1 - p_true_cal
    ncal = len(cal_idx)
    q_level = min(1.0, np.ceil((ncal + 1) * (1 - alpha)) / ncal)
    qhat = float(np.quantile(scores_cal, q_level))
    p_true_val = np.where(y[val_idx] == 1, p[val_idx], 1 - p[val_idx])
    covered = (1 - p_true_val) <= qhat
    coverage = float(covered.mean())
    set_sizes = []
    for pv in p[val_idx]:
        classes_in_set = sum(1 for cls_p in [pv, 1 - pv] if (1 - cls_p) <= qhat)
        set_sizes.append(classes_in_set)
    return {
        "alpha": alpha, "target_coverage": 1 - alpha, "qhat": round(qhat, 4),
        "n_calibration": int(ncal), "n_validation": int(n - ncal),
        "empirical_coverage": round(coverage, 4),
        "mean_set_size": round(float(np.mean(set_sizes)), 4),
        "pct_singleton_sets": round(float(np.mean(np.array(set_sizes) == 1)), 4),
    }

results = {}
oof_rows = []
for target_name in ["difficult", "easy"]:
    df = pd.read_csv(os.path.join(OUT_DIR, f"phase5_{target_name}_oof_per_site.csv"))
    y = df["y"].values
    p = df["proba"].values
    N = len(df)
    acc = float(np.nanmean((p >= 0.5).astype(int) == y))
    log(f"{target_name}: N={N} LOGO-cluster acc={acc:.4f} (matches Baseline_939_{target_name} from phase5_modeling_results.json)")

    brier, ece, reliability = brier_ece(y, p)
    iso = IsotonicRegression(out_of_bounds="clip")
    valid = ~np.isnan(p)
    iso.fit(p[valid], y[valid])
    p_cal = np.where(valid, iso.transform(np.nan_to_num(p)), np.nan)
    brier_cal, ece_cal, _ = brier_ece(y, p_cal)

    conformal = split_conformal(y, p)

    results[target_name] = {
        "N": N,
        "acc_logo_cluster": round(acc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "reliability_diagram": reliability,
        "isotonic_calibrated": {"brier_score": round(brier_cal, 4), "ece": round(ece_cal, 4)},
        "split_conformal_90pct": conformal,
    }
    for _, r in df.iterrows():
        oof_rows.append({"Locality_ID": r["Locality_ID"], "Region": r["Region"], "target": target_name,
                          "y_true": int(r["y"]), "oof_prob": round(float(r["proba"]), 4)})

pd.DataFrame(oof_rows).to_csv(os.path.join(OUT_DIR, "oof_probabilities_n939.csv"), index=False)
with open(os.path.join(OUT_DIR, "calibration_conformal_results_n939.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log("Wrote oof_probabilities_n939.csv + calibration_conformal_results_n939.json")

log("\nSummary (N=939):")
for target_name in ["difficult", "easy"]:
    r = results[target_name]
    print(f"  {target_name}: acc={r['acc_logo_cluster']} brier={r['brier_score']} ece={r['ece']} "
          f"-> isotonic brier={r['isotonic_calibrated']['brier_score']} ece={r['isotonic_calibrated']['ece']}")
    c = r["split_conformal_90pct"]
    print(f"    conformal: target={c['target_coverage']} empirical={c['empirical_coverage']} "
          f"mean_set_size={c['mean_set_size']} singleton_rate={c['pct_singleton_sets']}")
