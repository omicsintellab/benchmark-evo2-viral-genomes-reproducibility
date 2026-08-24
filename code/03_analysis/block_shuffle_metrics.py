#!/usr/bin/env python3
"""block_shuffle_metrics.py — metrics for the block-shuffling control (R1 3.3).

The `.npz` written by `06_revision_r1/01_block_shuffle.py` holds, for each genome, the
embedding of the original and of each shuffled condition. This script turns that into the two
quantities that answer the reviewer, and no others.

QUESTION 1 — does the embedding move when block order changes?
    Absolute displacement means nothing on its own: any perturbation moves an 8,192-dimensional
    vector by SOME amount. The number is only interpretable against a scale, and the relevant
    scale is the distance between DIFFERENT genomes, which is the distance the probe actually
    operates over. We therefore report displacement as a fraction of the typical between-genome
    distance (median pairwise distance among the originals).

    Reading: a small fraction means the embedding is essentially invariant to long-range
    rearrangement, and the claim must fall back to genome-level summary features. A fraction of
    order 1 means rearranging is like swapping genomes, and arrangement is represented.

QUESTION 2 — does the decodable information survive rearrangement?
    More direct and stronger than the distance: train the probe on the ORIGINAL embeddings and
    predict the target from the SHUFFLED ones. If R2 does not drop, what the probe reads is
    invariant to order, i.e. the probe was not using arrangement. This is what makes the answer
    to R1 3.3 a measurement rather than an opinion about distances.

CAVEAT THAT GOES INTO THE TEXT — stated here because it is a design limitation, not an
execution one: shuffling also shifts the 32-kb windows relative to the sequence, so part of the
measured displacement is a window-boundary effect rather than rearrangement. Without a circular
rotation condition (which preserves order and moves the windows the same way) the two cannot be
separated. The measured displacement is therefore an UPPER BOUND on the rearrangement effect,
which only strengthens the conclusion when it is small.

Usage:
    python block_shuffle_metrics.py --npz <file>.npz --features genome_features.tsv.gz \
        --out results/json/block_shuffle_metrics.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, wilcoxon

# the six primary targets
REG_FEATS = ["coding_fraction", "gene_density", "noncoding_bp", "n_genes",
             "mean_intergenic_len", "overlap_bp"]
ALPHAS = np.logspace(-2, 4, 13)


def log(m):
    print(m, flush=True)


def cos_dist(A, B):
    """Row-wise cosine distance."""
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.sum(a * b, axis=1)


def between_genome_scale(X, rng, n_pairs=20000):
    """Typical cosine distance between DIFFERENT genomes: the yardstick for displacement."""
    n = len(X)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    d = cos_dist(X[i[keep]], X[j[keep]])
    return float(np.median(d)), float(np.percentile(d, 5)), float(np.percentile(d, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="block_shuffle_metrics.json")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = np.load(a.npz, allow_pickle=True)
    accs = [str(x) for x in d["accs"]]
    conds = sorted((c for c in d.files if c.startswith("shuf") and not c.endswith("_idx")),
                   key=lambda c: int(c[4:]))
    X0 = d["original"].astype(np.float64)

    # Each condition may cover a subset of `accs` (a genome too short for that block count is
    # n_blocks). `<cond>_idx` diz a quais linhas de `original` ela corresponde. Arquivos
    # skipped for that condition only). Older .npz files without this field had all conditions
    # aligned and are treated as full coverage, so already published results still re-read.
    rows_of = {c: (d[f"{c}_idx"].astype(int) if f"{c}_idx" in d.files
                   else np.arange(len(accs))) for c in conds}
    log(f"genomes: {len(accs)} | conditions: original + {conds}")
    for c in conds:
        n = len(rows_of[c])
        if n < len(accs):
            log(f"  AVISO: {c} cobre {n}/{len(accs)} genomas — os ausentes eram curtos demais "
                f"for {c[4:]} blocks, which biases THAT condition towards long genomes")

    feats = pd.read_csv(a.features, sep="\t").set_index("accession")
    miss = [x for x in accs if x not in feats.index]
    log(f"CONTROLE POSITIVO — accessions no arquivo de features: "
        f"{len(accs)-len(miss)}/{len(accs)}")
    if miss:
        log(f"  WARNING: {len(miss)} missing, excluded from probe transfer")

    out = {"n_genomes": len(accs), "conditions": conds,
           # basename apenas: o caminho absoluto carrega o nome do usuario e do host
           # para dentro de um JSON que vai para repositorio publico
           "npz": os.path.basename(a.npz)}

    # ---- 1. displacement, against the between-genome yardstick
    med, p5, p95 = between_genome_scale(X0, rng)
    out["between_genome_cosine"] = {"median": med, "p5": p5, "p95": p95}
    log(f"\nbetween-genome yardstick: median cosine {med:.4f} (p5 {p5:.4f}, p95 {p95:.4f})")

    out["displacement"] = {}
    log("\n=== DESLOCAMENTO DO EMBEDDING ===")
    log(f"{'condition':>10} {'n':>5} {'median cos':>12} {'frac yardstick':>14} {'IQR':>18}")
    for c in conds:
        ridx = rows_of[c]
        dc = cos_dist(X0[ridx], d[c].astype(np.float64))
        frac = dc / med
        out["displacement"][c] = {
            "n_blocks": int(c[4:]),
            "n_genomes": int(len(ridx)),
            "cosine_median": float(np.median(dc)),
            "cosine_iqr": [float(np.percentile(dc, 25)), float(np.percentile(dc, 75))],
            "frac_between_genome_median": float(np.median(frac)),
            "frac_between_genome_iqr": [float(np.percentile(frac, 25)),
                                        float(np.percentile(frac, 75))],
        }
        log(f"{c:>10} {len(ridx):5d} {np.median(dc):12.4f} {np.median(frac):14.3f} "
            f"[{np.percentile(frac,25):.3f}; {np.percentile(frac,75):.3f}]")

    # dose-response: does displacement grow with the number of blocks?
    # The dose-response compares conditions WITH EACH OTHER, so it is only valid on the subset
    # they all cover: mixing populations would make the curve reflect sample composition.
    common = sorted(set.intersection(*[set(rows_of[c].tolist()) for c in conds])) if conds else []
    pos = {c: {int(g): k for k, g in enumerate(rows_of[c])} for c in conds}
    nb = np.array([int(c[4:]) for c in conds], dtype=float)
    per_genome = np.stack([
        cos_dist(X0[common], d[c].astype(np.float64)[[pos[c][g] for g in common]])
        for c in conds]) if common else np.zeros((len(conds), 0))
    log(f"\ndose-response computed on the subset common to all conditions: "
        f"{len(common)} genomas")
    rhos = [spearmanr(nb, per_genome[:, g]).statistic for g in range(per_genome.shape[1])]
    rhos = [r for r in rhos if np.isfinite(r)]
    try:
        w_p = float(wilcoxon(rhos).pvalue)
    except ValueError:
        w_p = float("nan")
    out["dose_response"] = {
        "spearman_rho_median": float(np.median(rhos)),
        "spearman_rho_iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
        "n_genomes_evaluated": len(rhos),
        "n_common_subset": int(len(common)),
        "wilcoxon_p_rho_ne_0": w_p,
        "note": "rho por genoma entre n_blocks e deslocamento; Wilcoxon dos rhos contra 0",
    }
    log(f"\ndose-resposta: rho mediano {np.median(rhos):+.3f} "
        f"[{np.percentile(rhos,25):+.3f}; {np.percentile(rhos,75):+.3f}] | Wilcoxon p={w_p:.3g}")

    # ---- 2. probe transfer: train on the original, predict on the shuffled
    keep = [i for i, x in enumerate(accs) if x in feats.index]
    sub = feats.loc[[accs[i] for i in keep]]
    groups = sub["family"].fillna("__sem_familia__").astype(str).values
    log("\n=== TRANSFERÊNCIA DO PROBE (treina no original, prediz no embaralhado) ===")
    log(f"{'alvo':>22} {'R2 original':>12} " + " ".join(f"{c:>9}" for c in conds))
    out["probe_transfer"] = {}
    for tgt in REG_FEATS:
        if tgt not in sub.columns:
            log(f"  pulando {tgt}: ausente no arquivo de features"); continue
        y = pd.to_numeric(sub[tgt], errors="coerce").values
        ok = np.isfinite(y)
        if ok.sum() < 30:
            log(f"  pulando {tgt}: n={ok.sum()} insuficiente"); continue
        idx = np.array(keep)[ok]
        yv, gv = y[ok], groups[ok]
        n_split = min(5, len(np.unique(gv)))
        if n_split < 2:
            log(f"  pulando {tgt}: grupos insuficientes"); continue
        gkf = GroupKFold(n_splits=n_split)
        Xo = X0[idx]
        preds = {c: np.full(len(idx), np.nan) for c in ["original"] + conds}
        for tr, te in gkf.split(Xo, yv, gv):
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            mdl.fit(Xo[tr], yv[tr])
            preds["original"][te] = mdl.predict(Xo[te])
            for c in conds:
                # `te` indexes `idx` (rows of X0); map to the rows of the condition,
                # leaving NaN where the condition does not cover the genome
                for j in te:
                    g = int(idx[j])
                    k = pos[c].get(g)
                    if k is not None:
                        preds[c][j] = mdl.predict(d[c].astype(np.float64)[k:k + 1])[0]
        def r2_of(v):
            ok2 = np.isfinite(v)
            if ok2.sum() < 10:
                return float("nan"), int(ok2.sum())
            yy = yv[ok2]
            sst = float(np.sum((yy - yy.mean()) ** 2))
            return float(1.0 - np.sum((yy - v[ok2]) ** 2) / sst), int(ok2.sum())
        r2n = {k: r2_of(v) for k, v in preds.items()}
        r2 = {k: v[0] for k, v in r2n.items()}
        out["probe_transfer"][tgt] = {
            "n": int(len(idx)), "n_groups": int(len(np.unique(gv))),
            "r2": r2, "n_per_condition": {k: v[1] for k, v in r2n.items()},
            "retention": {c: (float(r2[c] / r2["original"]) if r2["original"] > 0 else None)
                          for c in conds},
        }
        log(f"{tgt:>22} {r2['original']:12.3f} " + " ".join(f"{r2[c]:9.3f}" for c in conds))

    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
