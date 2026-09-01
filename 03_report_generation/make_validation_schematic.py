"""
Conceptual (non-geographic) schematic explaining cluster-aware LOGO
cross-validation vs. a naive random split, for readers unfamiliar with the
mechanics of spatial cross-validation. Synthetic illustrative points only
-- not real geosite data -- clearly a diagram, not a map.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.size": 9.5,
})

TRAIN_COL = "#A6A6A6"
TEST_COL = "#C9782E"
CLUSTER_COL = "#2B5F72"
LEAK_COL = "#B23A3A"

rng = np.random.default_rng(7)

# Two tight "near-duplicate" clusters + scattered singleton sites
cluster_a = np.array([0.5, 0.5]) + rng.normal(0, 0.06, size=(4, 2))
cluster_b = np.array([2.3, 1.6]) + rng.normal(0, 0.06, size=(3, 2))
singles = np.array([[1.2, 0.3], [3.0, 0.6], [1.7, 2.3], [3.4, 1.9], [0.3, 1.8], [2.6, 0.2]])

points = np.vstack([cluster_a, cluster_b, singles])
cluster_id = np.array([0]*4 + [1]*3 + list(range(2, 2 + len(singles))))

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))

# ---- Panel A: naive random split ----
ax = axes[0]
rng2 = np.random.default_rng(3)
naive_test = rng2.choice(len(points), size=3, replace=False)
for i, (x, y) in enumerate(points):
    is_test = i in naive_test
    color = TEST_COL if is_test else TRAIN_COL
    ax.scatter(x, y, s=70, facecolor=color, edgecolor="black", linewidth=0.6, zorder=3)
# highlight cluster A split across train/test
a_idx = list(range(4))
a_test = [i for i in a_idx if i in naive_test]
if a_test:
    circ = Circle(cluster_a.mean(axis=0), 0.22, fill=False, edgecolor=LEAK_COL,
                   linewidth=1.3, linestyle=(0, (3, 2)), zorder=2)
    ax.add_patch(circ)
    ax.annotate("near-duplicate pair\nsplit across train/test", xy=cluster_a.mean(axis=0),
                xytext=(0.55, -0.55), fontsize=7.2, color=LEAK_COL, ha="center",
                arrowprops=dict(arrowstyle="-", color=LEAK_COL, linewidth=0.8))
ax.set_title("Naive random split", fontsize=9.5, pad=6)
ax.set_xlim(-0.3, 3.9); ax.set_ylim(-0.9, 2.9)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# ---- Panel B: cluster-aware split ----
ax = axes[1]
held_out_clusters = {0}  # cluster A entirely held out
for i, (x, y) in enumerate(points):
    cid = cluster_id[i]
    is_test = cid in held_out_clusters
    color = TEST_COL if is_test else TRAIN_COL
    ax.scatter(x, y, s=70, facecolor=color, edgecolor="black", linewidth=0.6, zorder=3)
circ = Circle(cluster_a.mean(axis=0), 0.22, fill=False, edgecolor=CLUSTER_COL, linewidth=1.3, zorder=2)
ax.add_patch(circ)
ax.annotate("whole 500\\,m cluster\nheld out together", xy=cluster_a.mean(axis=0),
            xytext=(0.55, -0.55), fontsize=7.2, color=CLUSTER_COL, ha="center",
            arrowprops=dict(arrowstyle="-", color=CLUSTER_COL, linewidth=0.8))
ax.set_title("Cluster-aware split (this study)", fontsize=9.5, pad=6)
ax.set_xlim(-0.3, 3.9); ax.set_ylim(-0.9, 2.9)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=TRAIN_COL,
                   markeredgecolor="black", markersize=8, label="Training fold"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=TEST_COL,
                   markeredgecolor="black", markersize=8, label="Test fold")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "validation_schematic.pdf"))
plt.close(fig)
print("Saved validation_schematic.pdf")
