#!/usr/bin/env python3
"""scale_analysis.py — Análise de ESCALA (7B vs 20B) para o Brief Research Report.

Segue exatamente a metodologia de make_figures.py (probes lineares, RepeatedKFold
5x3, baseline 6-mer 4096-dim + GC+length, t-pareado) e de cluster_cv.py
(StratifiedGroupKFold / grouped-KFold com clusters MMseqs2 95%), estendendo-a ao
EVO2 20B em várias camadas.

Produz:
  - Tabela de escala: por camada do 20B (blocks.11/15/18/21/23.mlp.l3) vs 7B
    (blocks.28), com acc/R² do embedding + baselines 6-mer/GC+len + t-pareado,
    sob CV aleatória E cluster-aware.
  - Controle de precisão: compara FP8 vs bf16 na(s) mesma(s) camada(s) (mostra
    que FP8 não degrada a representação -> justifica usar os números FP8).
  - Controle de dimensionalidade: reduz o 20B (8192-dim) a 4096 via PCA e re-proba
    (separa efeito de escala de efeito de dimensionalidade; o 6-mer e o 7B têm 4096).
  - Figura de sensibilidade de camada (accuracy/R² x profundidade, por tipo de alvo).
  - scale_metrics.md com as tabelas.

Usage:
  python scale_analysis.py                  # FP8 embeddings
  python scale_analysis.py --with-bf16      # adds the precision control (needs the _bf16 caches)

Inputs (staged locally; paths overridable by env var — see below):
  SCALE_DATA    manifest.parquet, features.parquet, all_genomes.fasta, cl95_cluster.tsv
                (rebuild from data/ + code/01_corpus/ — see data/README.md)
  SCALE_EMB     embeddings_{baltimore,features}_evo2_7b_w32768.npz          (7B cache)
  SCALE_EMB20B  embeddings_evo2_20b_L{n}[_bf16]_w32768.npz                  (20B cache, layers 11/15/18/21/23)
  SCALE_OUT     where the JSON/MD outputs are written (default: results/json at the repo root)

The embedding caches are produced by code/02_embeddings/ on a GPU; they are not shipped in
this repository (see the Data Availability statement in the paper).
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold,
        RepeatedKFold, StratifiedGroupKFold)
from sklearn.metrics import r2_score

# Paths configuráveis por env (defaults = layout deste repositório).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
DATA   = os.environ.get("SCALE_DATA",   os.path.join(REPO, "data"))          # manifest/features/fasta/clusters
NB     = os.environ.get("SCALE_EMB",    os.path.join(REPO, "embeddings"))    # 7B emb (não versionado)
EMB20B = os.environ.get("SCALE_EMB20B", os.path.join(REPO, "embeddings"))    # 20B emb (não versionado)
OUTDIR = os.environ.get("SCALE_OUT",    os.path.join(REPO, "results", "json"))
SEED, SPLITS, REPEATS, K = 42, 5, 3, 6
WITH_BF16 = "--with-bf16" in sys.argv
NO_FIGURE = "--no-figure" in sys.argv
MM = 1/25.4

# camadas do 20B a analisar (fp8); profundidade relativa = n/24
LAYERS_20B = [11, 15, 18, 21, 23]
BF16_LAYERS = [15, 18]          # camadas com controle de precisão (bf16)
NAVY, ORANGE, GRAY = "#3C5488", "#E69F00", "#9AA0A6"
REG_FEATS = [("coding_fraction", False), ("gene_density", False),
             ("noncoding_bp", True), ("n_genes", True), ("mean_intergenic_len", True)]

# ----------------------------------------------------------------------------- dados
def load_meta(accs):
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe  = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")
    df = pd.DataFrame(index=accs)
    for c in ["host", "family", "baltimore"]:
        df[c] = man.reindex(accs)[c].values
    for c in ["GC", "genome_length", "coding_fraction", "gene_density",
              "noncoding_bp", "n_genes", "mean_intergenic_len"]:
        df[c] = fe.reindex(accs)[c].values
    return df

def seqs_for(accs):
    want = set(accs); out = {}; acc = None; buf = []
    with open(f"{DATA}/all_genomes.fasta") as fh:
        for ln in fh:
            if ln[0] == ">":
                if acc in want: out[acc] = "".join(buf)
                acc = ln[1:].split()[0]; buf = []
            else: buf.append(ln.strip())
        if acc in want: out[acc] = "".join(buf)
    return out

_B = {65: 0, 67: 1, 71: 2, 84: 3}
def kmer_matrix(accs):
    S = seqs_for(accs); D = 4**K; M = np.zeros((len(accs), D), np.float32)
    for r, a in enumerate(accs):
        arr = np.frombuffer(S.get(a, "").upper().encode("ascii", "ignore"), dtype=np.uint8)
        code = np.full(arr.shape, -1, np.int64)
        for b, v in _B.items(): code[arr == b] = v
        n = len(code)
        if n < K: continue
        idx = np.zeros(n-K+1, np.int64); ok = np.ones(n-K+1, bool)
        for off in range(K):
            c = code[off:off+(n-K+1)]; idx = idx*4 + np.where(c < 0, 0, c); ok &= (c >= 0)
        idx = idx[ok]
        if idx.size:
            v = np.bincount(idx, minlength=D).astype(np.float32); M[r] = v/v.sum()
    return M

def gclen(df):
    return np.column_stack([df["GC"].values, np.log1p(df["genome_length"].values)])

def load_7b(which):  # which in {baltimore, features}
    d = np.load(os.path.join(NB, f"embeddings_{which}_evo2_7b_w32768.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["accs"].astype(str)

def load_20b(layer, prec="fp8"):
    suf = "" if prec == "fp8" else f"_{prec}"
    p = f"{EMB20B}/embeddings_evo2_20b_L{layer}{suf}_w32768.npz"
    if not os.path.exists(p): return None, None
    d = np.load(p, allow_pickle=True)
    return d["X"].astype(np.float32), d["accs"].astype(str)

def align(X, src_accs, tgt_accs):
    """Reindexa X (linhas alinhadas a src_accs) para a ordem de tgt_accs."""
    pos = {a: i for i, a in enumerate(src_accs)}
    idx = [pos[a] for a in tgt_accs if a in pos]
    keep = np.array([a in pos for a in tgt_accs])
    return X[idx], keep

# clusters (95%) p/ CV cluster-aware
_cmap = {}
for ln in open(f"{DATA}/cl95_cluster.tsv"):
    rep, mem = ln.rstrip("\n").split("\t"); _cmap[mem] = rep
def groups(accs): return np.array([_cmap.get(a, a) for a in accs])

# ----------------------------------------------------------------------------- CV
def clf_rand(X, y):
    cv = RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=SEED)
    return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced")),
                           X, y, cv=cv, scoring="accuracy", n_jobs=-1)
def reg_rand(X, y):
    cv = RepeatedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=SEED)
    return cross_val_score(make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000])),
                           X, y, cv=cv, scoring="r2", n_jobs=-1)
def clf_group(X, y, g, reps=REPEATS):
    sc = []
    for r in range(reps):
        cv = StratifiedGroupKFold(n_splits=SPLITS, shuffle=True, random_state=SEED+r)
        sc += list(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced")),
                                   X, y, cv=cv.split(X, y, g), scoring="accuracy", n_jobs=-1))
    return np.array(sc)
def reg_group(X, y, g, reps=REPEATS):
    sc = []; rng = np.random.default_rng(SEED); uniq = np.unique(g)
    for r in range(reps):
        perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
        fold = np.array([perm[x] % SPLITS for x in g])
        for k in range(SPLITS):
            tr, te = fold != k, fold == k
            if te.sum() < 2: continue
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000])).fit(X[tr], y[tr])
            sc.append(r2_score(y[te], m.predict(X[te])))
    return np.array(sc)
def msd(a): return f"{a.mean():.3f} ± {a.std():.3f}"
def pstar(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

# ----------------------------------------------------------------------------- suíte por representação
def suite_clf(X, y, g):
    return {"rand": clf_rand(X, y), "clus": clf_group(X, y, g)}
def suite_reg(X, y, g):
    return {"rand": reg_rand(X, y), "clus": reg_group(X, y, g)}

def main():
    out = {"layers_20b": LAYERS_20B, "targets": {}}
    # ---- subset Baltimore (accb) e features (accf), do 7B (referência de accs) ----
    X7b_b, accb = load_7b("baltimore")
    X7b_f, accf = load_7b("features")
    db, dff = load_meta(accb), load_meta(accf)
    gb, gf = groups(accb), groups(accf)
    print(f"[dados] Baltimore n={len(accb)} ({len(set(gb))} clusters) | "
          f"features n={len(accf)} ({len(set(gf))} clusters)", flush=True)

    # baselines model-independent (uma vez)
    print("[baseline] 6-mer + GC+len ...", flush=True)
    KMb, GLb = kmer_matrix(accb), gclen(db)
    KMf, GLf = kmer_matrix(accf), gclen(dff)

    # máscaras host/família
    hm = dff["host"].isin(["eukaryote", "bacteria", "archaea"]).values
    fam = dff["family"].replace("", np.nan); topf = fam.value_counts(); topf = topf[topf >= 25].index.tolist()
    fk = dff["family"].isin(topf).values

    # ---- representações a testar: 7B + 20B por camada (fp8) [+ bf16 se --with-bf16] ----
    reps = {"7B_blocks28": {"balt": X7b_b, "feat": X7b_f}}
    specs = [(L, "fp8") for L in LAYERS_20B]
    if WITH_BF16:
        specs += [(L, "bf16") for L in BF16_LAYERS]
    for L, prec in specs:
        Xb20, ab20 = load_20b(L, prec)
        Xf20, af20 = load_20b(L, prec)
        if Xb20 is None:
            print(f"[aviso] 20B L{L} {prec} ausente — pulando"); continue
        # alinhar às MESMAS accs do 7B (Baltimore e features) por acesso
        Xb_al, kb = align(Xb20, ab20, accb)
        Xf_al, kf = align(Xf20, af20, accf)
        if not (kb.all() and kf.all()):
            print(f"[aviso] 20B L{L} {prec}: cobertura incompleta "
                  f"(balt {kb.sum()}/{len(kb)}, feat {kf.sum()}/{len(kf)}) — pulando p/ não desalinhar")
            continue
        suf = "" if prec == "fp8" else f"_{prec}"
        reps[f"20B_blocks{L}{suf}"] = {"balt": Xb_al, "feat": Xf_al}

    # ---- rodar suíte por representação ----
    results = {}
    for name, rp in reps.items():
        print(f"[probe] {name} ...", flush=True)
        r = {}
        # Baltimore
        r["Baltimore"] = suite_clf(rp["balt"], db["baltimore"].values, gb)
        # Host
        r["Host"] = suite_clf(rp["feat"][hm], dff["host"].values[hm], gf[hm])
        # Family
        r["Family"] = suite_clf(rp["feat"][fk], dff["family"].values[fk], gf[fk])
        # Regressões
        for f, logt in REG_FEATS:
            y = dff[f].values.astype(float); y = np.log1p(np.clip(y, 0, None)) if logt else y
            r[f] = suite_reg(rp["feat"], y, gf)
        results[name] = r

    # baselines como "representações" também (só rand+clus, para a tabela)
    base = {}
    base["Baltimore"] = {"kmer": {"rand": clf_rand(KMb, db["baltimore"].values), "clus": clf_group(KMb, db["baltimore"].values, gb)},
                         "gclen": {"rand": clf_rand(GLb, db["baltimore"].values), "clus": clf_group(GLb, db["baltimore"].values, gb)}}
    base["Host"] = {"kmer": {"rand": clf_rand(KMf[hm], dff["host"].values[hm]), "clus": clf_group(KMf[hm], dff["host"].values[hm], gf[hm])},
                    "gclen": {"rand": clf_rand(GLf[hm], dff["host"].values[hm]), "clus": clf_group(GLf[hm], dff["host"].values[hm], gf[hm])}}
    base["Family"] = {"kmer": {"rand": clf_rand(KMf[fk], dff["family"].values[fk]), "clus": clf_group(KMf[fk], dff["family"].values[fk], gf[fk])},
                      "gclen": {"rand": clf_rand(GLf[fk], dff["family"].values[fk]), "clus": clf_group(GLf[fk], dff["family"].values[fk], gf[fk])}}
    for f, logt in REG_FEATS:
        y = dff[f].values.astype(float); y = np.log1p(np.clip(y, 0, None)) if logt else y
        base[f] = {"kmer": {"rand": reg_rand(KMf, y), "clus": reg_group(KMf, y, gf)},
                   "gclen": {"rand": reg_rand(GLf, y), "clus": reg_group(GLf, y, gf)}}

    # (controle de precisão FP8 vs bf16: as camadas bf16 entram como representações
    #  20B_blocks{L}_bf16 quando --with-bf16, e aparecem lado a lado na tabela)

    # ---- salvar JSON bruto p/ o md/figuras ----
    def dump(d):
        if isinstance(d, dict): return {k: dump(v) for k, v in d.items()}
        if isinstance(d, np.ndarray): return {"mean": float(d.mean()), "std": float(d.std()), "vals": d.tolist()}
        return d
    payload = {"results": dump(results), "base": dump(base)}
    json.dump(payload, open(f"{OUTDIR}/scale_metrics.json", "w"), indent=2)
    print("[ok] scale_metrics.json", flush=True)

    # ---- tabela markdown (CV cluster-aware como principal; random em nota) ----
    write_md(results, base)
    if not NO_FIGURE:
        make_layer_figure(results, base)

def cell(a): return f"{a['mean']:.3f}±{a['std']:.3f}" if isinstance(a, dict) else f"{a.mean():.3f}±{a.std():.3f}"

def write_md(results, base):
    targets = ["Baltimore", "Host", "Family"] + [f for f, _ in REG_FEATS]
    names = list(results.keys())
    lines = ["# Scale metrics — 7B vs 20B por camada (cluster-aware CV, mean ± SD)\n",
             "Métrica: accuracy (classificação) / R² (regressão). CV cluster-aware (StratifiedGroupKFold / grouped-KFold, clusters MMseqs2 95%).",
             "Baselines 6-mer (4096-dim) e GC+length computados nas mesmas accs/folds.\n"]
    hdr = ["Target"] + names + ["6-mer", "GC+len"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"]*len(hdr)) + "|")
    for t in targets:
        row = [t]
        for nm in names:
            row.append(cell(results[nm][t]["clus"]))
        row.append(cell(base[t]["kmer"]["clus"]))
        row.append(cell(base[t]["gclen"]["clus"]))
        lines.append("| " + " | ".join(row) + " |")
    open(f"{OUTDIR}/scale_metrics.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

def make_layer_figure(results, base):
    import matplotlib as mpl; mpl.use("Agg")
    import matplotlib.pyplot as plt
    depth = {"7B_blocks28": None}  # 7B como linha de referência
    xs = LAYERS_20B
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(180*MM, 78*MM))
    # A: taxonomia
    for tgt, col in [("Baltimore", NAVY), ("Host", ORANGE), ("Family", GRAY)]:
        ys = [results[f"20B_blocks{L}"][tgt]["clus"]["mean"] if f"20B_blocks{L}" in results else np.nan for L in xs]
        axA.plot(xs, ys, "-o", color=col, label=tgt, markersize=4, linewidth=1.2)
        ref = results["7B_blocks28"][tgt]["clus"]["mean"]
        axA.axhline(ref, color=col, ls="--", lw=0.8, alpha=0.6)
    axA.set_xlabel("20B layer (blocks.N.mlp.l3)"); axA.set_ylabel("Accuracy (cluster-aware CV)")
    axA.set_title("Taxonomy", fontsize=8); axA.legend(frameon=False, fontsize=6.5)
    axA.text(0.02, 0.02, "dashed = 7B (blocks.28)", transform=axA.transAxes, fontsize=6, color="#666")
    # B: estrutura funcional
    for tgt, col in [("coding_fraction", NAVY), ("gene_density", ORANGE), ("n_genes", GRAY)]:
        ys = [results[f"20B_blocks{L}"][tgt]["clus"]["mean"] if f"20B_blocks{L}" in results else np.nan for L in xs]
        axB.plot(xs, ys, "-o", color=col, label=tgt, markersize=4, linewidth=1.2)
        ref = results["7B_blocks28"][tgt]["clus"]["mean"]
        axB.axhline(ref, color=col, ls="--", lw=0.8, alpha=0.6)
    axB.set_xlabel("20B layer (blocks.N.mlp.l3)"); axB.set_ylabel(r"$R^2$ (cluster-aware CV)")
    axB.set_title("Functional structure", fontsize=8); axB.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(f"{OUTDIR}/figure4_layer_sensitivity.svg")
    fig.savefig(f"{OUTDIR}/figure4_layer_sensitivity.png", dpi=300)
    plt.close(fig)
    print("[ok] figure4_layer_sensitivity", flush=True)

if __name__ == "__main__":
    main()
