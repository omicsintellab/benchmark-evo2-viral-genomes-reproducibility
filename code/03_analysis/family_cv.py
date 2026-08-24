#!/usr/bin/env python3
"""family_cv.py — family-grouped cross-validation and leave-one-family-out (revision R1).

Answers R1 3.2 and R2 #1/#2: clustering at 95% identity prevents near-duplicate leakage but
does NOT test generalisation to evolutionarily distinct viruses. This script repeats the main
analyses under taxonomic grouping and reports the schemes side by side, so the two questions
stay separate:

    random   — random cross-validation
    cl95     — groups = MMseqs2 clusters, 95% id / 85% cov (near-duplicate leakage)
    family   — groups = viral family (taxonomic non-independence)
    LOFO     — leave-one-family-out over families with n >= MIN_LOFO

Population: the records of the feature subset that have an assigned family. Records without one
cannot enter a family-grouped scheme and are dropped from the main analysis; the variant that
treats each of them as its own group is available with --with-singletons.

Grouping by family **subsumes** the per-organism grouping requested in R2 #4: segments of the
same virus share a family, so they always fall in the same fold. The script checks that
invariant and aborts if it does not hold.

Statistics (R1 3.5, R2 #6): fold scores of repeated cross-validation are not independent, and a
plain paired t-test understates the variance. We use the corrected resampled t-test of Nadeau
and Bengio, with Holm correction inside the primary set of six contrasts. cpg_oe and upa_oe are
a pre-declared negative control (dinucleotide composition — the 6-mer baseline is expected to
win) and stay OUT of the correction.

Usage:
    SCALE_DATA=<dir> SCALE_EMB20B=<dir> python code/03_analysis/family_cv.py [--out DIR]

Entradas: manifest.parquet, features.parquet, all_genomes.fasta, cl95_cluster.tsv,
embeddings_evo2_20b_L18_w32768.npz. Roda em CPU.
"""
import os, sys, json, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_analysis import (DATA, load_20b, align, kmer_matrix, gclen, seqs_for,
                            groups as cl95_groups, SEED, SPLITS, REPEATS)
from viral_features_extended import dinuc_oe

# PRIMARY (confirmatory) set: six contrasts, Evo 2 20B blocks.18 versus 6-mer.
PRIMARY = [("coding_fraction", False), ("gene_density", False), ("noncoding_bp", True),
           ("n_genes", True), ("mean_intergenic_len", True), ("overlap_bp", True)]
# Pre-declared negative control: dinucleotide composition. OUTSIDE the correction.
NEGCTRL = [("cpg_oe", False), ("upa_oe", False)]
MIN_LOFO = 30          # families with >= 30 genomes enter the leave-one-family-out
ALPHAS = [1, 10, 100, 1000]


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def reg_rand(X, y):
    cv = RepeatedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=SEED)
    return cross_val_score(ridge(), X, y, cv=cv, scoring="r2", n_jobs=-1)


def reg_group(X, y, g, reps=REPEATS):
    """Same mechanics as scale_analysis.reg_group: permute the groups and deal them
    round-robin across folds, so that a group is never split."""
    sc = []; rng = np.random.default_rng(SEED); uniq = np.unique(g)
    for r in range(reps):
        perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
        fold = np.array([perm[x] % SPLITS for x in g])
        for k in range(SPLITS):
            tr, te = fold != k, fold == k
            if te.sum() < 2 or tr.sum() < 10: continue
            m = ridge().fit(X[tr], y[tr])
            sc.append(r2_score(y[te], m.predict(X[te])))
    return np.array(sc)


def lofo(X, y, fam, fams):
    """Leave-one-family-out. Returns R² per family and the pooled R² over stacked
    out-of-fold predictions, which is more stable than averaging per-family R²
    when within-family variance is small."""
    per, yt, yp = {}, [], []
    for f in fams:
        te = fam == f; tr = ~te
        if te.sum() < 5: continue
        m = ridge().fit(X[tr], y[tr]); pred = m.predict(X[te])
        per[f] = float(r2_score(y[te], pred))
        yt.append(y[te]); yp.append(pred)
    pooled = float(r2_score(np.concatenate(yt), np.concatenate(yp))) if yt else float("nan")
    return per, pooled


def nadeau_bengio(a, b, n_splits=SPLITS):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for repeated-CV folds.
    Inflates the variance by (1/n + n_test/n_train) = (1/n + 1/(k-1))."""
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 2: return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var == 0: return float("nan"), float("nan")
    corr = 1.0 / n + 1.0 / (n_splits - 1)
    t = d.mean() / np.sqrt(corr * var)
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def boot_ci(a, b, n=10000, seed=SEED):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(pvals, names):
    """Holm-Bonferroni. Returns {name: adjusted_p}."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return {names[i]: adj[i] for i in range(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "..", "results", "json"))
    ap.add_argument("--with-singletons", action="store_true",
                    help="Include records without a family as singleton groups (sensitivity analysis).")
    a = ap.parse_args()

    # ---- 1. population -------------------------------------------------------
    X20, acc20 = load_20b(18, "fp8")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")
    md = pd.DataFrame(index=accs)
    md["family"] = man.reindex(accs)["family"].replace("", np.nan).values
    md["organism"] = fe.reindex(accs)["organism"].values
    for c, _ in PRIMARY:
        md[c] = fe.reindex(accs)[c].values

    has_fam = md["family"].notna().values
    print(f"[family_cv] records com embedding: {len(accs)}")
    print(f"[family_cv] with family: {has_fam.sum()} | without family (dropped): {(~has_fam).sum()}")

    if a.with_singletons:
        md["family"] = md["family"].fillna(pd.Series(accs, index=md.index))
        keep = np.ones(len(accs), bool)
        print("[family_cv] --with-singletons: records without a family become their own groups")
    else:
        keep = has_fam

    accs_k = accs[keep]; md = md[keep]
    X20_k = X20.astype(np.float32)[keep]
    fam = md["family"].values.astype(str)

    # R2 #4 invariant: family subsumes organism
    bad = md.groupby("organism")["family"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        print(f"[ERROR] {len(bad)} organisms map to more than one family; grouping by family "
              f"would not protect segments. Examples: {list(bad.index[:5])}")
        sys.exit(1)
    nseg = (md.groupby("organism").size() > 1).sum()
    print(f"[family_cv] invariant OK: family subsumes organism "
          f"({nseg} multi-record organisms always land in the same fold)")

    # ---- 2. representations --------------------------------------------------
    print("[family_cv] montando 6-mer e GC+len ...", flush=True)
    KM = kmer_matrix(accs_k.tolist())
    md["GC"] = fe.reindex(accs_k)["GC"].values
    md["genome_length"] = fe.reindex(accs_k)["genome_length"].values
    GL = gclen(md)

    print("[family_cv] CpG/UpA o/e (negative control) ...", flush=True)
    seqs = seqs_for(accs_k.tolist())
    cu = [dinuc_oe(seqs.get(x, "")) for x in accs_k]
    md["cpg_oe"] = [c for c, _ in cu]; md["upa_oe"] = [u for _, u in cu]

    reps = {"evo2_20b_blocks18": X20_k, "6mer": KM, "gc_len": GL}
    g_cl95 = cl95_groups(accs_k)

    famsize = pd.Series(fam).value_counts()
    lofo_fams = famsize[famsize >= MIN_LOFO].index.tolist()
    print(f"[family_cv] families: {famsize.size} | with >= {MIN_LOFO}: {len(lofo_fams)} "
          f"({famsize[famsize >= MIN_LOFO].sum()} records)")

    # ---- 3. CV --------------------------------------------------------------
    out = {"population": {"n_with_embedding": int(len(accs)),
                          "n_with_family": int(has_fam.sum()),
                          "n_without_family": int((~has_fam).sum()),
                          "n_analysed": int(keep.sum()),
                          "with_singletons": bool(a.with_singletons),
                          "n_families": int(famsize.size),
                          "n_multirecord_organisms": int(nseg)},
           "config": {"splits": SPLITS, "repeats": REPEATS, "seed": SEED,
                      "alphas": ALPHAS, "min_lofo": MIN_LOFO,
                      "lofo_families": {f: int(famsize[f]) for f in lofo_fams}},
           "targets": {}}

    for name, logt in PRIMARY + NEGCTRL:
        y = md[name].values.astype(float)
        m = np.isfinite(y)
        yv = np.log1p(np.clip(y[m], 0, None)) if logt else y[m]
        rec = {"log1p": logt, "n": int(m.sum()), "primary": (name, logt) in PRIMARY, "reps": {}}
        print(f"\n[family_cv] === {name} (n={m.sum()}, log1p={logt}) ===", flush=True)
        for rname, R in reps.items():
            Xr = R[m]
            s_rand = reg_rand(Xr, yv)
            s_cl95 = reg_group(Xr, yv, g_cl95[m])
            s_fam = reg_group(Xr, yv, fam[m])
            per, pooled = lofo(Xr, yv, fam[m], lofo_fams)
            rec["reps"][rname] = {
                "random": {"mean": float(s_rand.mean()), "std": float(s_rand.std()),
                           "scores": s_rand.tolist()},
                "cl95": {"mean": float(s_cl95.mean()), "std": float(s_cl95.std()),
                         "scores": s_cl95.tolist()},
                "family": {"mean": float(s_fam.mean()), "std": float(s_fam.std()),
                           "scores": s_fam.tolist()},
                "lofo": {"pooled_r2": pooled, "per_family": per},
            }
            print(f"  {rname:>20}  rand={s_rand.mean():.3f}  cl95={s_cl95.mean():.3f}  "
                  f"fam={s_fam.mean():.3f}  LOFO={pooled:.3f}", flush=True)
        out["targets"][name] = rec

    # ---- 4. statistics: Evo 2 versus 6-mer under family-grouped CV ----------
    prim = [n for n, _ in PRIMARY]
    tests, pv = {}, []
    for name in prim:
        e = out["targets"][name]["reps"]["evo2_20b_blocks18"]["family"]["scores"]
        k = out["targets"][name]["reps"]["6mer"]["family"]["scores"]
        t, p = nadeau_bengio(e, k)
        lo, hi = boot_ci(e, k)
        tests[name] = {"delta_mean": float(np.mean(e) - np.mean(k)), "ci95": [lo, hi],
                       "t_nadeau_bengio": t, "p_raw": p, "n_folds": len(e)}
        pv.append(p)
    adj = holm(pv, prim)
    for name in prim:
        tests[name]["p_holm"] = adj[name]
    out["primary_tests"] = {"scheme": "family-grouped CV", "contrast": "evo2_20b_blocks18 vs 6mer",
                            "correction": "holm", "tests": tests}

    # negative control, no correction
    neg = {}
    for name, _ in NEGCTRL:
        e = out["targets"][name]["reps"]["evo2_20b_blocks18"]["family"]["scores"]
        k = out["targets"][name]["reps"]["6mer"]["family"]["scores"]
        t, p = nadeau_bengio(e, k); lo, hi = boot_ci(e, k)
        neg[name] = {"delta_mean": float(np.mean(e) - np.mean(k)), "ci95": [lo, hi],
                     "t_nadeau_bengio": t, "p_raw": p}
    out["negative_control"] = {"note": "pre-declared; 6-mer expected to win; outside the correction",
                               "tests": neg}

    os.makedirs(a.out, exist_ok=True)
    suf = "_singletons" if a.with_singletons else ""
    dst = os.path.join(a.out, f"family_cv_metrics{suf}.json")
    with open(dst, "w") as fh: json.dump(out, fh, indent=1)
    print(f"\n[family_cv] -> {dst}")


if __name__ == "__main__":
    main()
