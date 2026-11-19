#!/usr/bin/env python3
"""alpha_length_r3.py — two controls requested in the third round of review.

R3 #10 (ridge penalty). The paper selects the ridge penalty inside each training fold by
generalised cross-validation over the grid {1, 10, 100, 1000}. With 8,192 standardised
features and ~1,350 training rows that grid may be too low: if the selected penalty piles
up at the upper boundary, the probes are under-regularised. This script reports the
distribution of the selected penalty per fold on the paper's grid and on an extended grid
(10^-1 to 10^6), together with the R² each grid yields, so the effect on the reported
numbers is explicit.

R3 #3 (genome length). Genome length is never reported as a probe target, and the
reviewer notes a mechanical route from pooling to length: genomes up to 32,768 bp are
embedded in a single forward pass, longer ones are averaged over up to eight windows, so
the pooled vector could encode window count and therefore size. This script decodes
log genome length and repeats it on the single-window subset (<= 32,768 bp), where the
number of pooled windows is constant at one and that route cannot operate.

Population and folds are those of the inferential analysis (family_cv.py): records of the
feature subset with an assigned family, under the primary identity-clustered scheme (cl95)
and the family-grouped scheme; seed 42, five folds, three repeats. The fold mechanics,
the cluster map and the embedding loader are imported from the published scripts rather
than reimplemented, and a positive control checks that this script reproduces a published
number before anything else is reported.

Usage:
    SCALE_DATA=<dir> SCALE_EMB20B=<dir> python code/07_revision_r3/alpha_length_r3.py

Inputs: manifest.parquet, features.parquet, cl95_cluster.tsv, embeddings_evo2_20b_L18_w32768.npz.
No FASTA is needed: the compositional baselines are not re-run here. Runs on CPU.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_analysis"))
from scale_analysis import (DATA, load_20b, groups as cl95_groups,  # noqa: E402
                            SEED, SPLITS, REPEATS)

# Same six targets as the primary inferential set, plus genome length (new in R3).
PRIMARY = [("coding_fraction", False), ("gene_density", False), ("noncoding_bp", True),
           ("n_genes", True), ("mean_intergenic_len", True), ("overlap_bp", True)]
LENGTH = ("genome_length", True)

GRIDS = {"paper": [1.0, 10.0, 100.0, 1000.0],
         "wide": np.logspace(-1, 6, 15).tolist()}      # 0.1 ... 1e6, ~2 per decade
WINDOW_BP = 32768                                      # single forward pass at or below this

# Published value used as the positive control: final_statistics.json -> primary.gene_density.r2_a
# (Evo 2 20B blocks.18, cl95 scheme, paper grid, 1,691 records with a family).
CONTROL = {"target": "gene_density", "scheme": "cl95", "grid": "paper", "expected": 0.7756179415883562}


def ridge(alphas):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))


def reg_group_alphas(X, y, g, alphas, reps=REPEATS):
    """Grouped repeated CV with the mechanics of scale_analysis.reg_group (groups permuted
    and dealt round-robin across folds, so a group is never split), additionally recording
    the penalty selected inside each training fold."""
    sc, al = [], []
    rng = np.random.default_rng(SEED)
    uniq = np.unique(g)
    for _ in range(reps):
        perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
        fold = np.array([perm[x] % SPLITS for x in g])
        for k in range(SPLITS):
            tr, te = fold != k, fold == k
            if te.sum() < 2 or tr.sum() < 10:
                continue
            m = ridge(alphas).fit(X[tr], y[tr])
            sc.append(r2_score(y[te], m.predict(X[te])))
            al.append(float(m[-1].alpha_))
    return np.array(sc), np.array(al)


def summarise(scores, alphas, grid):
    top = max(grid)
    cnt = {}
    for a in alphas:
        cnt[f"{a:g}"] = cnt.get(f"{a:g}", 0) + 1
    return {"r2_mean": float(scores.mean()), "r2_std": float(scores.std()),
            "n_folds": int(len(scores)),
            "alpha_counts": cnt,
            "alpha_median": float(np.median(alphas)),
            "alpha_min": float(alphas.min()), "alpha_max": float(alphas.max()),
            "frac_at_grid_max": float(np.mean(alphas >= top)),
            "grid_max": float(top)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "json",
                                                  "alpha_length_r3.json"))
    a = ap.parse_args()

    # ---- population: identical to family_cv.py ------------------------------
    X20, acc20 = load_20b(18, "fp8")
    if X20 is None:
        sys.exit("[erro] cache de embeddings do 20B L18 nao encontrado (SCALE_EMB20B)")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")

    md = pd.DataFrame(index=accs)
    md["family"] = man.reindex(accs)["family"].replace("", np.nan).values
    for c, _ in PRIMARY + [LENGTH]:
        md[c] = fe.reindex(accs)[c].values

    keep = md["family"].notna().values
    accs_k = accs[keep]
    md = md[keep]
    X = X20.astype(np.float32)[keep]
    fam = md["family"].values.astype(str)
    g_cl95 = cl95_groups(accs_k)
    length = md["genome_length"].values.astype(float)
    single = length <= WINDOW_BP

    print(f"[r3] records com embedding: {len(accs)} | com familia (analisados): {keep.sum()}")
    print(f"[r3] uma janela (<= {WINDOW_BP} bp): {int(single.sum())} "
          f"({100*single.mean():.1f}%) | multi-janela: {int((~single).sum())}")
    print(f"[r3] features do embedding: {X.shape[1]} | grades: "
          f"paper={GRIDS['paper']} wide=10^-1..10^6 ({len(GRIDS['wide'])} pontos)", flush=True)

    schemes = {"cl95": g_cl95, "family": fam}
    out = {"design": {
               "population": {"n_with_embedding": int(len(accs)), "n_analysed": int(keep.sum()),
                              "n_single_window": int(single.sum()),
                              "n_multi_window": int((~single).sum()),
                              "window_bp": WINDOW_BP},
               "cv": {"splits": SPLITS, "repeats": REPEATS, "seed": SEED,
                      "schemes": ["cl95 (primary, identity-clustered)", "family (grouped)"]},
               "representation": "evo2_20b blocks.18.mlp.l3, FP8, mean-pooled",
               "grids": {k: ([float(x) for x in v] if k == "paper" else
                             {"from": 1e-1, "to": 1e6, "n": len(v)}) for k, v in GRIDS.items()},
               "note": "No compositional baseline is re-run here; the question is the penalty "
                       "of the 8,192-dimensional embedding probe and the decodability of length.",
           },
           "positive_control": {}, "alpha": {}, "length": {}}

    # ---- positive control ----------------------------------------------------
    y = md[CONTROL["target"]].values.astype(float)
    m = np.isfinite(y)
    sc, al = reg_group_alphas(X[m], y[m], schemes[CONTROL["scheme"]][m], GRIDS[CONTROL["grid"]])
    got = float(sc.mean())
    ok = abs(got - CONTROL["expected"]) < 1e-6
    out["positive_control"] = {**CONTROL, "obtained": got, "reproduced": bool(ok)}
    print(f"\n[r3] controle positivo {CONTROL['target']}/{CONTROL['scheme']}: "
          f"obtido {got:.6f} | publicado {CONTROL['expected']:.6f} | "
          f"{'OK' if ok else 'DIVERGE'}", flush=True)
    if not ok:
        sys.exit("[erro] a mecanica de folds nao reproduz o numero publicado; nao reportar nada.")

    # ---- R3 #10: penalty distribution on the six primary targets -------------
    print("\n[r3] === #10 penalidade selecionada por fold ===", flush=True)
    for name, logt in PRIMARY:
        y = md[name].values.astype(float)
        m = np.isfinite(y)
        yv = np.log1p(np.clip(y[m], 0, None)) if logt else y[m]
        rec = {"log1p": logt, "n": int(m.sum()), "schemes": {}}
        for sname, g in schemes.items():
            srec = {}
            for gname, grid in GRIDS.items():
                sc, al = reg_group_alphas(X[m], yv, g[m], grid)
                srec[gname] = summarise(sc, al, grid)
            srec["delta_r2_wide_minus_paper"] = (srec["wide"]["r2_mean"]
                                                 - srec["paper"]["r2_mean"])
            rec["schemes"][sname] = srec
            print(f"  {name:>22} {sname:>7}  paper R²={srec['paper']['r2_mean']:.3f} "
                  f"(alpha med={srec['paper']['alpha_median']:g}, "
                  f"{100*srec['paper']['frac_at_grid_max']:.0f}% no teto)  "
                  f"wide R²={srec['wide']['r2_mean']:.3f} "
                  f"(alpha med={srec['wide']['alpha_median']:g})  "
                  f"Δ={srec['delta_r2_wide_minus_paper']:+.3f}", flush=True)
        out["alpha"][name] = rec

    # ---- R3 #3: genome length as a target ------------------------------------
    print("\n[r3] === #3 log genome length como alvo ===", flush=True)
    name, logt = LENGTH
    yl = np.log1p(length)
    for pop, mask in (("all", np.ones(len(yl), bool)), ("single_window", single)):
        rec = {"n": int(mask.sum()), "schemes": {}}
        for sname, g in schemes.items():
            srec = {}
            for gname, grid in GRIDS.items():
                sc, al = reg_group_alphas(X[mask], yl[mask], g[mask], grid)
                srec[gname] = summarise(sc, al, grid)
            rec["schemes"][sname] = srec
            print(f"  log_length [{pop:>13}] n={mask.sum():>4} {sname:>7}  "
                  f"paper R²={srec['paper']['r2_mean']:.3f}  "
                  f"wide R²={srec['wide']['r2_mean']:.3f} "
                  f"(alpha med={srec['wide']['alpha_median']:g})", flush=True)
        out["length"][pop] = rec
    out["length"]["note"] = ("The GC-and-length control is degenerate for this target "
                             "(log length is one of its two inputs) and is not reported. "
                             "single_window = records up to one 32,768-bp forward pass, "
                             "where the number of pooled windows is constant.")

    dst = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[r3] -> {dst}")


if __name__ == "__main__":
    main()
