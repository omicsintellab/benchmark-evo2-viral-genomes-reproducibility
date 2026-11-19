#!/usr/bin/env python3
"""contrast_wide_grid_r3.py — the six primary contrasts re-tested with an extended ridge grid (R3 #10).

alpha_length_r3.py shows that the ridge penalty of the embedding probe saturates the
published grid {1, 10, 100, 1000}: the upper boundary is selected in essentially every
fold. Re-tuning only the embedding would answer half the question, because the 6-mer
baseline may be under-regularised too. This script re-runs both representations on an
extended grid (10^-1 to 10^6) and re-tests the six primary contrasts with the same
corrected resampled t-test and Holm correction as the paper, under both cross-validation
schemes, so the effect on the *contrast* — not just on one side of it — is explicit.

The paper's grid is re-run here as well and checked against the published deltas as a
positive control before the extended-grid numbers are reported.

Usage:
    SCALE_DATA=<dir> SCALE_EMB20B=<dir> python code/07_revision_r3/contrast_wide_grid_r3.py

Inputs: manifest.parquet, features.parquet, all_genomes.fasta, cl95_cluster.tsv,
embeddings_evo2_20b_L18_w32768.npz. Runs on CPU.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_analysis"))
sys.path.insert(0, HERE)
from scale_analysis import (DATA, load_20b, kmer_matrix, gclen,  # noqa: E402
                            groups as cl95_groups, SEED, SPLITS, REPEATS)
from family_cv import nadeau_bengio, nb_ci, holm, PRIMARY  # noqa: E402
from alpha_length_r3 import GRIDS, reg_group_alphas, summarise  # noqa: E402

# Published deltas (final_statistics.json -> primary[*].delta), cl95 scheme, paper grid.
CONTROL_DELTAS = {"coding_fraction": 0.2951073653532446, "gene_density": 0.37488088341643183}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "json",
                                                  "contrast_wide_grid_r3.json"))
    a = ap.parse_args()

    X20, acc20 = load_20b(18, "fp8")
    if X20 is None:
        sys.exit("[erro] cache de embeddings do 20B L18 nao encontrado (SCALE_EMB20B)")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")

    md = pd.DataFrame(index=accs)
    md["family"] = man.reindex(accs)["family"].replace("", np.nan).values
    for c, _ in PRIMARY:
        md[c] = fe.reindex(accs)[c].values
    keep = md["family"].notna().values
    accs_k = accs[keep]; md = md[keep]
    X = X20.astype(np.float32)[keep]
    fam = md["family"].values.astype(str)
    g_cl95 = cl95_groups(accs_k)
    print(f"[r3] analisados: {keep.sum()} | dims embedding: {X.shape[1]}", flush=True)

    print("[r3] montando 6-mer ...", flush=True)
    KM = kmer_matrix(accs_k.tolist())
    md["GC"] = fe.reindex(accs_k)["GC"].values
    md["genome_length"] = fe.reindex(accs_k)["genome_length"].values
    GL = gclen(md)
    reps = {"evo2_20b_blocks18": X, "6mer": KM, "gc_len": GL}
    schemes = {"cl95": g_cl95, "family": fam}

    out = {"design": {"population": int(keep.sum()),
                      "cv": {"splits": SPLITS, "repeats": REPEATS, "seed": SEED},
                      "grids": {"paper": GRIDS["paper"],
                                "wide": {"from": 1e-1, "to": 1e6, "n": len(GRIDS["wide"])}},
                      "test": "corrected resampled t-test (Nadeau & Bengio, 2003), "
                              "Holm within the primary set of six",
                      "contrast": "evo2_20b_blocks18 vs 6mer"},
           "scores": {}, "contrasts": {}}

    # ---- fold scores for every (target, representation, scheme, grid) -------
    for name, logt in PRIMARY:
        y = md[name].values.astype(float)
        m = np.isfinite(y)
        yv = np.log1p(np.clip(y[m], 0, None)) if logt else y[m]
        out["scores"][name] = {"log1p": logt, "n": int(m.sum()), "reps": {}}
        for rname, R in reps.items():
            Xr = R[m]
            rec = {}
            for sname, g in schemes.items():
                for gname, grid in GRIDS.items():
                    sc, al = reg_group_alphas(Xr, yv, g[m], grid)
                    rec[f"{sname}|{gname}"] = {**summarise(sc, al, grid),
                                               "scores": sc.tolist()}
            out["scores"][name]["reps"][rname] = rec
            print(f"  {name:>22} {rname:>18}  "
                  + "  ".join(f"{k}={rec[k]['r2_mean']:.3f}" for k in rec), flush=True)

    # ---- contrasts, per scheme and grid -------------------------------------
    for sname in schemes:
        for gname in GRIDS:
            key = f"{sname}|{gname}"
            tests, pv, names = {}, [], []
            for name, _ in PRIMARY:
                e = out["scores"][name]["reps"]["evo2_20b_blocks18"][key]["scores"]
                k = out["scores"][name]["reps"]["6mer"][key]["scores"]
                t, p = nadeau_bengio(e, k)
                lo, hi = nb_ci(e, k)
                tests[name] = {"r2_evo2": float(np.mean(e)), "r2_6mer": float(np.mean(k)),
                               "delta": float(np.mean(e) - np.mean(k)), "ci95": [lo, hi],
                               "t_nadeau_bengio": t, "p_raw": p, "n_folds": len(e)}
                pv.append(p); names.append(name)
            adj = holm(pv, names)
            for name in names:
                tests[name]["p_holm"] = adj[name]
                tests[name]["significant_holm"] = bool(adj[name] < 0.05)
            out["contrasts"][key] = tests

    # ---- positive control: paper grid must reproduce the published deltas ---
    ctrl = {}
    for name, exp in CONTROL_DELTAS.items():
        got = out["contrasts"]["cl95|paper"][name]["delta"]
        ctrl[name] = {"expected": exp, "obtained": got, "reproduced": bool(abs(got - exp) < 1e-6)}
        print(f"[r3] controle {name}: obtido {got:.6f} | publicado {exp:.6f} | "
              f"{'OK' if ctrl[name]['reproduced'] else 'DIVERGE'}")
    out["positive_control"] = ctrl

    print("\n[r3] === contraste por esquema e grade (Holm) ===")
    for key, tests in out["contrasts"].items():
        n_sig = sum(v["significant_holm"] for v in tests.values())
        print(f"  {key:>14}: {n_sig}/6 significativos | "
              + " ".join(f"{n}:{v['delta']:+.3f}" for n, v in tests.items()))

    dst = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[r3] -> {dst}")


if __name__ == "__main__":
    main()
