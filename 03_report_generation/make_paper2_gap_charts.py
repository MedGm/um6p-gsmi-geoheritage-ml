"""
03_report_generation/make_paper2_gap_charts.py  (2026-08-23)

Paper 2's professional redesign of the old leave_region_out.pdf style:
horizontal grouped bars, model accuracy vs. local-majority baseline, gap
labeled directly. Two separate figures per the approved design plan --
single-region training only, and merged/multi-region training only -- with
the single-region figure's caption explaining why the three thinnest
regions (Guelmim-Oued Noun N=7, Laâyoune-Sakia El Hamra N=15,
Casablanca-Settat N=7) are absent from it.

Output: report/figures/paper2_gap_chart_single.pdf
        report/figures/paper2_gap_chart_merged.pdf
"""
import os, json
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "report", "figures")

EASY_COL, DIFFICULT_COL, MAJ_COL, ACCENT = "#76A5AF", "#C1650A", "#555555", "#2B5F72"

plt.rcParams.update({"font.family": "serif", "font.size": 9})

def make_chart(rows, out_name, title):
    """rows: list of (label, target, acc, maj) ordered top-to-bottom."""
    labels = [r[0] for r in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * n + 1.0))
    y = np.arange(n)[::-1]
    for yi, (label, target, acc, maj) in zip(y, rows):
        color = DIFFICULT_COL if target == "Difficult" else EASY_COL
        ax.barh(yi, acc, height=0.6, color=color, zorder=3)
        ax.plot([maj, maj], [yi - 0.32, yi + 0.32], color=MAJ_COL, linewidth=1.6, zorder=4)
        gap = (acc - maj) * 100
        gap_txt = f"{gap:+.1f} pp"
        text_x = max(acc, maj) + 0.02
        ax.text(text_x, yi, gap_txt, va="center", ha="left", fontsize=7.5,
                color=("#1a7a1a" if gap >= 0 else "#a01515"), fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("Accuracy (LOGO-cluster CV)")
    ax.set_title(title, fontsize=10, fontweight="bold", color=ACCENT, loc="left")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    handles = [
        plt.Rectangle((0,0),1,1, color=DIFFICULT_COL, label="Difficult-vs-not accuracy"),
        plt.Rectangle((0,0),1,1, color=EASY_COL, label="Easy-vs-not accuracy"),
        plt.Line2D([0],[0], color=MAJ_COL, linewidth=1.6, label="Local majority-class baseline"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16 if n < 10 else -0.09),
              ncol=3, fontsize=7.5, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(OUT, out_name)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Single-region training only (7 individually-modeled regions x 2 targets)
# ---------------------------------------------------------------------------
single = json.load(open(os.path.join(FW, "results/json/training/phase5_paper2_best_feature_results.json")))
ORDER = ["Fés-Meknés", "Béni Mellal-Khénifra", "Tanger-Tétouan-Al Hoceima", "Drâa-Tafilalet",
         "Souss-Massa", "Marrakech-Safi", "Eddakhla-Oued Eddahab"]
by_region = {(e["region"], e["target"]): e for e in single}
rows_single = []
for region in ORDER:
    for target in ["Difficult", "Easy"]:
        e = by_region[(region, target)]
        label = f"{region} ({'D' if target=='Difficult' else 'E'})"
        rows_single.append((label, target, e["best_acc"], e["local_majority"]))
make_chart(rows_single, "paper2_gap_chart_single.pdf",
           "Single-region training: accuracy vs. local majority baseline")

# ---------------------------------------------------------------------------
# Merged/multi-region training (3 merged groups x 2 targets)
# ---------------------------------------------------------------------------
merged = json.load(open(os.path.join(FW, "results/json/training/phase5_paper2_merged_regions_results.json")))
GROUP_LABELS = {
    "South_GuelmimLaayoune": "Guelmim + Laâyoune",
    "South_GuelmimLaayouneEddakhla": "Guelmim + Laâyoune + Eddakhla",
    "RabatCasablanca": "Rabat-Salé-Kénitra + Casablanca-Settat",
}
by_group = {(e["group"], e["target"]): e for e in merged}
rows_merged = []
for group in ["South_GuelmimLaayoune", "South_GuelmimLaayouneEddakhla", "RabatCasablanca"]:
    for target in ["Difficult", "Easy"]:
        e = by_group[(group, target)]
        label = f"{GROUP_LABELS[group]} ({'D' if target=='Difficult' else 'E'})"
        rows_merged.append((label, target, e["best_acc"], e["local_majority"]))
make_chart(rows_merged, "paper2_gap_chart_merged.pdf",
           "Merged/multi-region training: accuracy vs. local majority baseline")
