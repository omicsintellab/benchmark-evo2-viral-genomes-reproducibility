#!/usr/bin/env python3
"""Painel combinado Figura 1 (2x2) — fusão das antigas Figuras 1 e 2 do Brief
Research Report, para caber no limite de 4 Figuras/Tabelas da Frontiers in
Bioinformatics (Brief Research Report).

Reusa a MESMA lógica/dados de make_figures_20b.py (nenhum recômputo de CV):
  A = confusion matrix Baltimore (ex-Fig.1a)
  B = accuracy Baltimore/Host/Family vs baselines (ex-Fig.1b)
  C = R^2 das 8 features vs baselines (ex-Fig.2a)
  D = PCA do embedding colorido por coding fraction (ex-Fig.2b)

Saída: figure1_combined.svg/png
"""
import os, json, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
NAVY, ORANGE, GRAY = "#3C5488", "#E69F00", "#9AA0A6"
SEQ = "cividis"
BNAME = {"I": "I dsDNA", "II": "II ssDNA", "III": "III dsRNA", "IV": "IV ssRNA+",
         "V": "V ssRNA−", "VI": "VI ssRNA-RT", "VII": "VII dsDNA-RT"}


def plab(ax, s):
    ax.text(-0.16, 1.10, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def clean(ax):
    ax.tick_params(length=3, width=0.7)


def load(name):
    return json.load(open(os.path.join(DATA_DIR, name)))


def vals(d):
    return np.array(d["vals"])


scale = load("scale_metrics.json")
vfe = load("viral_features_extended_metrics.json")
art = load("fig_artifacts_20b.json")

REP20 = scale["results"]["20B_blocks18"]
BASE = scale["base"]

CLF_TARGETS = ["Baltimore", "Host", "Family"]
REG_TARGETS = ["coding_fraction", "gene_density", "noncoding_bp", "n_genes", "mean_intergenic_len"]
VFE_TARGETS = ["cpg_oe", "upa_oe", "overlap_bp"]


def get_20b_meanstd(target, cv="clus"):
    if target in VFE_TARGETS:
        d = vfe["results"]["20B_blocks18"][target]
        return d["mean"], d["std"]
    d = REP20[target][cv]
    return d["mean"], d["std"]


def get_base_meanstd(target, which, cv="clus"):
    if target in VFE_TARGETS:
        key = "6mer" if which == "kmer" else "GC+len"
        d = vfe["results"][key][target]
        return d["mean"], d["std"]
    d = BASE[target][which][cv]
    return d["mean"], d["std"]


def grouped(ax, cats, ylabel, ylim, get_mean_fn, legend=True):
    x = np.arange(len(cats)); w = 0.26
    keys = [("emb", "Evo 2 20B embedding (blocks.18)", NAVY), ("kmer", "6-mer composition", ORANGE), ("gclen", "GC + length", GRAY)]
    for i, (k, lab, c) in enumerate(keys):
        m, s = [], []
        for t in cats:
            mm, ss = get_mean_fn(t, k)
            m.append(mm); s.append(ss)
        ax.bar(x + (i - 1) * w, m, w, yerr=s, color=c, edgecolor="none", label=lab,
               error_kw=dict(elinewidth=0.7, capsize=2, capthick=0.7, ecolor="#333333"))
    ax.set_xticks(x); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); clean(ax)
    if legend:
        ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.1, columnspacing=1.4, borderaxespad=0.6, fontsize=6.3)
    return x


def mean_fn(t, which):
    if which == "emb":
        return get_20b_meanstd(t)
    return get_base_meanstd(t, which)


def figure1_combined():
    fig = plt.figure(figsize=(180 * MM, 155 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.32,
                           top=0.93, bottom=0.07)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A — confusion matrix (Baltimore, ex-Fig.1a)
    cm = art["cm"]; L = cm["labels"]; M = np.array(cm["M"])
    im = axA.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="equal")
    axA.set_xticks(range(len(L))); axA.set_yticks(range(len(L)))
    axA.set_xticklabels(L); axA.set_yticklabels([BNAME[b] for b in L])
    axA.set_xlabel("Predicted (Baltimore group)"); axA.set_ylabel("True class")
    for i in range(len(L)):
        for j in range(len(L)):
            v = M[i, j]
            axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                     color="black" if v > 0.55 else "white")
    axA.set_title(f"Baltimore class, Evo 2 20B (acc = {cm['acc']:.2f})", fontsize=8)
    axA.tick_params(length=0)
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, labelsize=6.5); cb.set_label("row fraction", fontsize=6.5)
    plab(axA, "a")

    # B — classification accuracy vs baselines (ex-Fig.1b)
    grouped(axB, CLF_TARGETS, "Accuracy (cluster-aware CV)", (0, 1.0), mean_fn, legend=False)
    axB.set_xticklabels(["Baltimore", "Host", "Family"])
    plab(axB, "b")

    # C — R^2 per feature vs baselines (ex-Fig.2a)
    feats = REG_TARGETS + VFE_TARGETS
    flab = ["coding\nfrac.", "gene\ndens.", "noncod.\nbp", "gene\ncount",
            "interg.\nlen", "CpG\no/e", "UpA\no/e", "overlap\n(log)"]
    grouped(axC, feats, r"$R^2$", (0, 1.0), mean_fn, legend=False)
    axC.set_xticklabels(flab, fontsize=6.2)
    axC.axvline(4.5, color="#cccccc", lw=0.6, ls=(0, (2, 2)), zorder=0)
    axC.text(2.0, -0.24, "genomic architecture", ha="center", fontsize=6, color="#666666", transform=axC.get_xaxis_transform())
    axC.text(6.0, -0.24, "sequence composition", ha="center", fontsize=6, color="#666666", transform=axC.get_xaxis_transform())
    plab(axC, "c")

    # D — PCA scatter colored by coding fraction (ex-Fig.2b)
    PC1, PC2, cf = np.array(art["pca"]["PC1"]), np.array(art["pca"]["PC2"]), np.array(art["pca"]["coding_fraction"])
    sc = axD.scatter(PC1, PC2, c=cf, cmap=SEQ, s=8, edgecolor="none", alpha=0.85, vmin=0, vmax=1)
    axD.set_xlabel("PC1"); axD.set_ylabel("PC2")
    axD.set_title(f"Evo 2 20B embedding PCA (n = {art['pca']['n']})", fontsize=8); clean(axD)
    cb = fig.colorbar(sc, ax=axD, fraction=0.046, pad=0.04); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, labelsize=6.5); cb.set_label("coding fraction", fontsize=6.5)
    plab(axD, "d")

    handles = [Patch(color=NAVY, label="Evo 2 20B embedding (blocks.18)"),
               Patch(color=ORANGE, label="6-mer composition"),
               Patch(color=GRAY, label="GC + length")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.1, columnspacing=1.6, fontsize=7.5)

    fig.savefig(f"{OUT_DIR}/figure1_combined.svg")
    fig.savefig(f"{OUT_DIR}/figure1_combined.png", dpi=300)
    plt.close(fig)
    print("figure1_combined")


if __name__ == "__main__":
    figure1_combined()
