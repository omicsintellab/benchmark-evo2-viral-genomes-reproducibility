#!/usr/bin/env python3
"""Figura 3 combinada (A/B) — sensibilidade de camada + escala 20B vs 7B.

Antes a Figura 3 era só o heatmap de camadas (metade da Seção 3.4). Este painel
cobre a seção inteira:
  A = layer sensitivity heatmap do 20B (blocks 11..23) — justifica blocks.18
  B = comparação de escala 20B vs 7B por alvo (cluster-aware), com marcadores
      da versão com dimensionalidade casada (150-D PCA) sobrepostos, tornando
      visual a afirmação "advantage persists at matched dimensionality".

Reusa os MESMOS JSON cacheados (scale_metrics.json, pca_control_metrics.json),
sem recomputar CV. Saída: figure3_combined.svg/png.
"""
import os, json, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
# Metric JSONs shipped in the repo (results/json/). Override with FIG20B_DATA to point at a
# fresh run of code/03_analysis/.
DATA_DIR = os.environ.get("FIG20B_DATA", os.path.join(REPO, "results", "json"))
# Figures are written to figures/ at the repo root.
OUT_DIR = os.environ.get("FIG20B_OUT", os.path.join(REPO, "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out", "axes.titlepad": 6,
    "savefig.bbox": "tight", "svg.fonttype": "none", "figure.dpi": 150,
})
NAVY, GREEN, ORANGE = "#3C5488", "#009E73", "#E69F00"
SEQ = "cividis"


def plab(ax, s):
    ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def load(name):
    return json.load(open(os.path.join(DATA_DIR, name)))


scale = load("scale_metrics.json")
pca = load("pca_control_metrics.json")

TARGETS = ["Baltimore", "Host", "Family", "coding_fraction", "gene_density",
           "noncoding_bp", "n_genes", "mean_intergenic_len"]
TLAB = ["Baltimore", "Host", "Family", "coding\nfraction", "gene\ndensity",
        "noncoding\nbp", "gene\ncount", "interg.\nlen"]


def figure3_combined():
    fig = plt.figure(figsize=(180 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.28,
                          top=0.86, bottom=0.20, left=0.10, right=0.97)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- A: layer sensitivity heatmap (20B) ----
    layers = [11, 15, 18, 21, 23]
    M = np.zeros((len(TARGETS), len(layers)))
    for i, t in enumerate(TARGETS):
        for j, L in enumerate(layers):
            M[i, j] = scale["results"][f"20B_blocks{L}"][t]["clus"]["mean"]
    im = axA.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    axA.set_xticks(range(len(layers))); axA.set_xticklabels([f"{L}" for L in layers])
    axA.set_xlabel("layer (blocks.N)")
    axA.set_yticks(range(len(TARGETS))); axA.set_yticklabels(TLAB, fontsize=6.5)
    for i in range(len(TARGETS)):
        for j in range(len(layers)):
            v = M[i, j]
            axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                     color="black" if v > 0.55 else "white")
    j18 = layers.index(18)
    axA.add_patch(plt.Rectangle((j18 - 0.5, -0.5), 1, len(TARGETS), fill=False, edgecolor=ORANGE, lw=1.8))
    axA.set_title("Layer sensitivity, Evo 2 20B", fontsize=8)
    axA.tick_params(length=0)
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, labelsize=6.5); cb.set_label("accuracy / R²", fontsize=6.5)
    plab(axA, "a")

    # ---- B: scale 20B vs 7B (native bars + matched-dim markers) ----
    x = np.arange(len(TARGETS)); w = 0.36
    v20 = np.array([scale["results"]["20B_blocks18"][t]["clus"]["mean"] for t in TARGETS])
    s20 = np.array([scale["results"]["20B_blocks18"][t]["clus"]["std"] for t in TARGETS])
    v7 = np.array([scale["results"]["7B_blocks28"][t]["clus"]["mean"] for t in TARGETS])
    s7 = np.array([scale["results"]["7B_blocks28"][t]["clus"]["std"] for t in TARGETS])
    p20 = np.array([pca["reps"]["20B_blocks18"][t]["clus_pca"]["mean"] for t in TARGETS])
    p7 = np.array([pca["reps"]["7B_blocks28"][t]["clus_pca"]["mean"] for t in TARGETS])

    axB.bar(x - w / 2, v20, w, yerr=s20, color=NAVY, edgecolor="none", label="Evo 2 20B (blocks.18)",
            error_kw=dict(elinewidth=0.7, capsize=2, capthick=0.7, ecolor="#333333"))
    axB.bar(x + w / 2, v7, w, yerr=s7, color=GREEN, edgecolor="none", label="Evo 2 7B (blocks.28)",
            error_kw=dict(elinewidth=0.7, capsize=2, capthick=0.7, ecolor="#333333"))
    # matched-dimensionality (150-D PCA) markers
    axB.scatter(x - w / 2, p20, marker="D", s=13, facecolor="white", edgecolor=NAVY, linewidth=0.9, zorder=5)
    axB.scatter(x + w / 2, p7, marker="D", s=13, facecolor="white", edgecolor=GREEN, linewidth=0.9, zorder=5)

    axB.set_xticks(x); axB.set_xticklabels(TLAB, fontsize=6.2)
    axB.set_ylim(0, 1.05); axB.set_ylabel("accuracy / R² (cluster-aware CV)")
    axB.tick_params(length=3, width=0.7)
    plab(axB, "b")

    handles = [Patch(color=NAVY, label="Evo 2 20B (blocks.18)"),
               Patch(color=GREEN, label="Evo 2 7B (blocks.28)"),
               Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                      markeredgecolor="#555555", markersize=5, label="matched dim. (150-D PCA)")]
    axB.legend(handles=handles, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.1, columnspacing=1.3,
               borderaxespad=0.4, fontsize=6.3)

    fig.savefig(f"{OUT_DIR}/figure3_combined.svg")
    fig.savefig(f"{OUT_DIR}/figure3_combined.png", dpi=300)
    plt.close(fig)
    print("figure3_combined")


if __name__ == "__main__":
    figure3_combined()
