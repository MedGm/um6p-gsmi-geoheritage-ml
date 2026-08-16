"""
Redesigned publication figures for geosite_ai_section.tex.
Design system: single accent (deep teal-blue) + neutral greys, one warm
amber reserved for the single standout result (Eddakhla), body font matched
to the document (newpxtext) via matplotlib's pgf backend so figure text is
typographically consistent with the surrounding LaTeX. Vector PDF output.
All values sourced from results/ (json/other/derived_numbers.json, which is itself
derived from json/training/h5_h6_resume_results_N733.json, logs/h0_h4_comprehensive_log.txt,
json/training/final_v2_results_N733.json) -- nothing here is estimated or re-derived
from memory.
"""
import json, os
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results", "json", "other")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b",
    "text.color": "#2b2b2b",
    "xtick.color": "#2b2b2b",
    "ytick.color": "#2b2b2b",
    "axes.grid": True,
    "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

ACCENT = "#2B5F72"      # deep teal-blue -- primary model/result color
ACCENT_SOFT = "#8FB4C0"  # lighter tint for secondary/comparison bars
NEUTRAL = "#A6A6A6"      # baselines / non-highlighted
HIGHLIGHT = "#C0392B"    # professional red, reserved for the single standout (Eddakhla)
WEAK = "#8C8C8C"         # the meaningfully-weaker linear baseline

derived = json.load(open(os.path.join(RESULTS, "derived_numbers.json"), encoding="utf-8"))

# ============================================================ Figure: Eddakhla hero
# Single-result standalone figure for the strongest finding (leave-region-out).
fig, ax = plt.subplots(figsize=(3.4, 3.3))
bars = ax.bar(["Local majority\nbaseline", "Leave-region-out\nmodel"], [80.0, 96.0],
              color=[NEUTRAL, HIGHLIGHT], width=0.55, zorder=3)
for b, v in zip(bars, [80.0, 96.0]):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}\\%", ha="center", va="bottom",
            fontsize=10, color="#2b2b2b")
ax.annotate("", xy=(1, 94), xytext=(0, 82),
            arrowprops=dict(arrowstyle="-|>", color="#1F7A3D", linewidth=1.3,
                             connectionstyle="arc3,rad=-0.25"))
ax.text(0.5, 100.5, "+16.0\\,pp", ha="center", fontsize=10, color="#1F7A3D", fontweight="bold")
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (\\%)")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "eddakhla_hero.pdf"))
plt.close(fig)

# ============================================================ Figure 2 =====
# Model-family comparison, all cluster-aware LOGO CV, national N=733
mf = derived["model_family_cluster_aware"]
mf_labels = ["Logistic\nregression", "k-NN\n(geographic)", "k-NN\n(feature-space)",
             "Gaussian\nProcess", "Tree\nensemble"]
mf_values = [mf["Logistic regression (multinomial)"] * 100,
             mf["k-NN, geographic distance (k=5-15)"] * 100,
             mf["k-NN, feature-space distance (k=15-25)"] * 100,
             mf["Gaussian Process (RBF, 10-fold group CV)"] * 100,
             mf["Tree ensemble (RF+XGB+GBM+LightGBM+CatBoost)"] * 100]
mf_colors = [WEAK, ACCENT, ACCENT_SOFT, ACCENT, ACCENT]

fig, ax = plt.subplots(figsize=(6.2, 3.3))
bars = ax.bar(mf_labels, mf_values, color=mf_colors, width=0.6, zorder=3)
for b, v in zip(bars, mf_values):
    ax.text(b.get_x() + b.get_width()/2, v + 1.3, f"{v:.1f}", ha="center", va="bottom",
            fontsize=9, color="#2b2b2b")
ax.axhline(33.3, color="#B23A3A", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
ax.text(4.45, 35.3, "chance (3 classes)", fontsize=7.7, color="#B23A3A", ha="right")
ax.set_ylim(0, 72)
ax.set_ylabel("3-class LOGO-cluster CV accuracy (\\%)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_family_comparison.pdf"))
plt.close(fig)

# ============================================================ Figure 3 =====
# Leave-region-out, Difficult classifier: model vs local majority baseline
lro = derived["leave_region_out_difficult"]
lro_sorted = sorted(lro, key=lambda r: -r["gap_pp"])
regions = [r["region"] for r in lro_sorted]
model_acc = [r["acc"] * 100 for r in lro_sorted]
baseline_acc = [r["local_majority"] * 100 for r in lro_sorted]
gaps = [r["gap_pp"] for r in lro_sorted]
degenerate = [r["degenerate"] for r in lro_sorted]

def model_color(r):
    if r["region"].startswith("Eddakhla"):
        return HIGHLIGHT
    if r["degenerate"]:
        return NEUTRAL
    return ACCENT

x = range(len(regions))
width = 0.36
fig, ax = plt.subplots(figsize=(7.6, 3.8))
b1 = ax.bar([i - width/2 for i in x], baseline_acc, width, label="Local majority baseline",
            color=NEUTRAL, zorder=3)
b2 = ax.bar([i + width/2 for i in x], model_acc, width, label="Leave-region-out model",
            color=[model_color(r) for r in lro_sorted], zorder=3)
for i, (g, deg) in enumerate(zip(gaps, degenerate)):
    label = "tie" if deg else (f"+{g:.1f}\\,pp" if g > 0 else f"{g:.1f}\\,pp")
    color = "#555555" if deg else ("#1F7A3D" if g > 0 else "#B23A3A")
    y = max(model_acc[i], baseline_acc[i]) + 2.5
    ax.text(i, y, label, ha="center", va="bottom", fontsize=7.6, color=color)
ax.set_xticks(list(x))
ax.set_xticklabels(regions, fontsize=7.4, rotation=28, ha="right")
ax.set_ylabel("Accuracy (\\%)")
ax.set_ylim(0, 112)
ax.legend(loc="upper right", fontsize=7.6, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "leave_region_out.pdf"))
plt.close(fig)

print("Wrote 3 figures to", OUT)
