#!/usr/bin/env python3
"""Figuras + métricas do Brief Research Report — versão 20B (protagonista).

Lê os JSONs já computados pelas etapas anteriores (scale_analysis.py,
viral_features_extended.py, fig_artifacts_20b.py, generation_completion_20b.py) —
não recomputa CV, só reformata/plota. Modelo principal: Evo 2 20B, blocks.18, FP8.

Entrada (JSONs pequenos, versionados em results/json/; ver DATA_DIR):
  scale_metrics.json, viral_features_extended_metrics.json,
  fig_artifacts_20b.json, generation_summary_evo2_20b.json,
  pca_control_metrics.json, precision_control_metrics.json (nota metodológica)

Saídas: figure1_20b.svg/png (Baltimore CM + classificação), figure2_20b.svg/png
(R² 8 features + PCA scatter), figure3_20b.svg/png (geração/completação),
figure4_layer_sensitivity.svg/png (varredura de camadas, transparência da
escolha blocks.18), probe_metrics_20b.md (tabela com testes pareados).
"""
import os, sys, json, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
# Metric JSONs shipped in the repo (results/json/). Override with FIG20B_DATA to point at a
# fresh run of code/03_analysis/.
DATA_DIR = os.environ.get("FIG20B_DATA", os.path.join(REPO, "results", "json"))
# Figures are written to figures/ at the repo root.
OUT_DIR = os.environ.get("FIG20B_OUT", os.path.join(REPO, "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
DOC = os.path.join(REPO, "results", "tables")
MM = 1/25.4

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
BALT = ["I", "II", "III", "IV", "V", "VI", "VII"]
BNAME = {"I": "I dsDNA", "II": "II ssDNA", "III": "III dsRNA", "IV": "IV ssRNA+",
         "V": "V ssRNA−", "VI": "VI ssRNA-RT", "VII": "VII dsDNA-RT"}

def plab(ax, s): ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
def clean(ax): ax.tick_params(length=3, width=0.7)

def load(name):
    return json.load(open(os.path.join(DATA_DIR, name)))

def vals(d): return np.array(d["vals"])

# ----------------------------------------------------------------- dados
scale = load("scale_metrics.json")
vfe = load("viral_features_extended_metrics.json")
art = load("fig_artifacts_20b.json")
gen = load("generation_summary_evo2_20b.json")

REP20 = scale["results"]["20B_blocks18"]
BASE = scale["base"]

CLF_TARGETS = ["Baltimore", "Host", "Family"]
REG_TARGETS = ["coding_fraction", "gene_density", "noncoding_bp", "n_genes", "mean_intergenic_len"]
VFE_TARGETS = ["cpg_oe", "upa_oe", "overlap_bp"]

def get_20b(target, cv="clus"):
    if target in VFE_TARGETS:
        return np.array([vfe["results"]["20B_blocks18"][target]["mean"]] * 1)  # sem "vals" no vfe -> só mean/std
    return vals(REP20[target][cv])

def get_20b_meanstd(target, cv="clus"):
    if target in VFE_TARGETS:
        d = vfe["results"]["20B_blocks18"][target]; return d["mean"], d["std"]
    d = REP20[target][cv]; return d["mean"], d["std"]

def get_base_meanstd(target, which, cv="clus"):
    if target in VFE_TARGETS:
        key = "6mer" if which == "kmer" else "GC+len"
        d = vfe["results"][key][target]; return d["mean"], d["std"]
    d = BASE[target][which][cv]; return d["mean"], d["std"]

def get_base_vals(target, which, cv="clus"):
    if target in VFE_TARGETS:
        return None
    return vals(BASE[target][which][cv])

# ----------------------------------------------------------------- grouped bar
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

def mean_fn_clf(t, which):
    if which == "emb": return get_20b_meanstd(t)
    return get_base_meanstd(t, which)

def mean_fn_reg(t, which):
    if which == "emb": return get_20b_meanstd(t)
    return get_base_meanstd(t, which)

# ======================= FIGURE 1: Baltimore CM + classificacao =======================
def figure1():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(180*MM, 78*MM), gridspec_kw={"width_ratios": [1.05, 1.25]})
    cm = art["cm"]; L = cm["labels"]; M = np.array(cm["M"])
    im = axA.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="equal")
    axA.set_xticks(range(len(L))); axA.set_yticks(range(len(L)))
    axA.set_xticklabels(L); axA.set_yticklabels([BNAME[b] for b in L])
    axA.set_xlabel("Predicted (Baltimore group)"); axA.set_ylabel("True class")
    for i in range(len(L)):
        for j in range(len(L)):
            v = M[i, j]; axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                                   color="black" if v > 0.55 else "white")
    axA.set_title(f"Baltimore class, Evo 2 20B (acc = {cm['acc']:.2f})", fontsize=8)
    axA.tick_params(length=0)
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, labelsize=6.5); cb.set_label("row fraction", fontsize=6.5)
    plab(axA, "a")
    grouped(axB, CLF_TARGETS, "Accuracy (cluster-aware CV)", (0, 1.12), mean_fn_clf)
    axB.set_xticklabels(["Baltimore", "Host", "Family"])
    plab(axB, "b")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(f"{OUT_DIR}/figure1_20b.svg"); fig.savefig(f"{OUT_DIR}/figure1_20b.png", dpi=300); plt.close(fig)
    print("figure1_20b")

# ======================= FIGURE 2: R2 (8 features) + PCA =======================
def figure2():
    feats = REG_TARGETS + VFE_TARGETS
    flab = ["coding\nfrac.", "gene\ndens.", "noncod.\nbp", "gene\ncount",
            "interg.\nlen", "CpG\no/e", "UpA\no/e", "overlap\n(log)"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(205*MM, 82*MM), gridspec_kw={"width_ratios": [1.6, 1]})
    grouped(axA, feats, r"$R^2$", (0, 1.0), mean_fn_reg)
    axA.set_xticklabels(flab, fontsize=6.2)
    axA.axvline(4.5, color="#cccccc", lw=0.6, ls=(0, (2, 2)), zorder=0)
    axA.text(2.0, -0.26, "genomic architecture", ha="center", fontsize=6, color="#666666", transform=axA.get_xaxis_transform())
    axA.text(6.0, -0.26, "sequence composition", ha="center", fontsize=6, color="#666666", transform=axA.get_xaxis_transform())
    plab(axA, "a")
    PC1, PC2, cf = np.array(art["pca"]["PC1"]), np.array(art["pca"]["PC2"]), np.array(art["pca"]["coding_fraction"])
    sc = axB.scatter(PC1, PC2, c=cf, cmap=SEQ, s=8, edgecolor="none", alpha=0.85, vmin=0, vmax=1)
    axB.set_xlabel("PC1"); axB.set_ylabel("PC2")
    axB.set_title(f"Evo 2 20B embedding PCA (n = {art['pca']['n']})", fontsize=8); clean(axB)
    cb = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, labelsize=6.5); cb.set_label("coding fraction", fontsize=6.5)
    plab(axB, "b")
    fig.tight_layout(w_pad=2.4)
    fig.savefig(f"{OUT_DIR}/figure2_20b.svg"); fig.savefig(f"{OUT_DIR}/figure2_20b.png", dpi=300); plt.close(fig)
    print("figure2_20b")

# ======================= FIGURE 3: geracao/completacao =======================
def figure3():
    p = gen["perplexity"]; c = gen["completion"]
    perp = {"euk": (p["euk_heldout"]["mean"], p["euk_heldout"]["std"]),
            "phg": (p["phage_seen"]["mean"], p["phage_seen"]["std"])}
    bits = {"euk": (c["bits_evo"]["euk_heldout"], c["bits_markov"]["euk_heldout"]),
            "phg": (c["bits_evo"]["phage_seen"], c["bits_markov"]["phage_seen"])}
    kmer = {"euk": (c["kmer_evo"]["euk_heldout"], c["kmer_markov"]["euk_heldout"]),
            "phg": (c["kmer_evo"]["phage_seen"], c["kmer_markov"]["phage_seen"])}
    slab = ["Eukaryotic\nviruses\n(held-out)", "Bacterio-\nphages\n(seen)"]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(180*MM, 68*MM)); x = np.arange(2)
    axA.bar(x, [perp[s][0] for s in ["euk", "phg"]], 0.5, yerr=[perp[s][1] for s in ["euk", "phg"]],
            color=[ORANGE, NAVY], edgecolor="none", error_kw=dict(elinewidth=0.7, capsize=2.5, ecolor="#333333"))
    axA.set_xticks(x); axA.set_xticklabels(slab); axA.set_ylabel("bits / nucleotide")
    axA.set_title("Genome perplexity, Evo 2 20B", fontsize=8); clean(axA); plab(axA, "a")
    w = 0.34
    for ax, dat, ylab, ttl, lbl in [(axB, bits, "gap bits / nt", "Completion: likelihood", "b"),
                                     (axC, kmer, "k-mer cosine (gen vs real)", "Completion: composition", "c")]:
        ax.bar(x - w/2, [dat[s][0] for s in ["euk", "phg"]], w, color=NAVY, edgecolor="none", label="Evo 2 20B")
        ax.bar(x + w/2, [dat[s][1] for s in ["euk", "phg"]], w, color=GRAY, edgecolor="none", label="Markov-4")
        ax.set_xticks(x); ax.set_xticklabels(slab); ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=8)
        ax.legend(frameon=False, loc="upper right", handlelength=1.1); clean(ax); plab(ax, lbl)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(f"{OUT_DIR}/figure3_20b.svg"); fig.savefig(f"{OUT_DIR}/figure3_20b.png", dpi=300); plt.close(fig)
    print("figure3_20b")

# ======================= FIGURE 4: sensibilidade a camada (transparencia) =======================
def figure4():
    layers = [11, 15, 18, 21, 23]
    targets = CLF_TARGETS + REG_TARGETS
    tlab = ["Baltimore", "Host", "Family", "coding\nfraction", "gene\ndensity",
            "noncoding\nbp", "gene\ncount", "interg.\nlen"]
    M = np.zeros((len(targets), len(layers)))
    for i, t in enumerate(targets):
        for j, L in enumerate(layers):
            M[i, j] = scale["results"][f"20B_blocks{L}"][t]["clus"]["mean"]
    fig, ax = plt.subplots(figsize=(120*MM, 78*MM))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers))); ax.set_xticklabels([f"blocks.{L}" for L in layers])
    ax.set_yticks(range(len(targets))); ax.set_yticklabels(tlab, fontsize=6.5)
    for i in range(len(targets)):
        for j in range(len(layers)):
            v = M[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                  color="black" if v > 0.55 else "white")
    j18 = layers.index(18)
    ax.add_patch(plt.Rectangle((j18-0.5, -0.5), 1, len(targets), fill=False, edgecolor=ORANGE, lw=1.8))
    ax.set_title("Layer sensitivity, Evo 2 20B (cluster-aware CV)", fontsize=8)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, labelsize=6.5); cb.set_label("accuracy / R²", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure4_layer_sensitivity.svg"); fig.savefig(f"{OUT_DIR}/figure4_layer_sensitivity.png", dpi=300)
    plt.close(fig)
    print("figure4_layer_sensitivity")

# ======================= tabela =======================
def pstar(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
def msd(m, s): return f"{m:.3f} ± {s:.3f}"

def write_table():
    hdr = ["Probe", "Target", "Evo 2 20B (blocks.18)", "6-mer composition", "GC + length",
           "20B vs k-mer", "20B vs GC+len"]
    rows = []
    for t in CLF_TARGETS:
        eo, es = get_20b_meanstd(t); ko, ks = get_base_meanstd(t, "kmer"); go, gs = get_base_meanstd(t, "gclen")
        ev, kv, gv = vals(REP20[t]["clus"]), get_base_vals(t, "kmer"), get_base_vals(t, "gclen")
        rows.append(["Class. (acc)", t, msd(eo, es), msd(ko, ks), msd(go, gs),
                     pstar(ttest_rel(ev, kv).pvalue), pstar(ttest_rel(ev, gv).pvalue)])
    for t in REG_TARGETS:
        eo, es = get_20b_meanstd(t); ko, ks = get_base_meanstd(t, "kmer"); go, gs = get_base_meanstd(t, "gclen")
        ev, kv, gv = vals(REP20[t]["clus"]), get_base_vals(t, "kmer"), get_base_vals(t, "gclen")
        rows.append(["Regr. (R²)", t, msd(eo, es), msd(ko, ks), msd(go, gs),
                     pstar(ttest_rel(ev, kv).pvalue), pstar(ttest_rel(ev, gv).pvalue)])
    for t in VFE_TARGETS:
        eo, es = get_20b_meanstd(t); ko, ks = get_base_meanstd(t, "kmer"); go, gs = get_base_meanstd(t, "gclen")
        rows.append(["Regr. (R²)", t, msd(eo, es), msd(ko, ks), msd(go, gs), "n/a¹", "n/a¹"])
    out = ["# Probe metrics — Evo 2 20B (blocks.18, FP8), cluster-aware CV (5×3 repeats, mean ± SD)\n",
           "Significance: paired t-test across the 15 fold values; *** p<0.001, ** p<0.01, * p<0.05, ns.\n",
           "1. CpG/UpA/overlap probes were run with only mean/SD cached (no per-fold significance test).\n",
           "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows: out.append("| " + " | ".join(str(c) for c in r) + " |")
    open(f"{DOC}/probe_metrics_20b.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))

if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4(); write_table()
    print("OK")
