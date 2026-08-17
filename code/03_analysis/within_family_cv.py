#!/usr/bin/env python3
"""within_family_cv.py — CV DENTRO de cada família (revisão R1 3.2).

Complementa family_cv.py. O leave-one-family-out mede duas coisas ao mesmo tempo e
não as separa: (a) existe sinal de arquitetura dentro de uma família? e (b) esse
sinal transfere de uma família para outra? Um R² baixo no LOFO pode ser ausência de
sinal intra-família OU apenas restrição de amplitude — dentro de uma família a
variação do alvo é muito menor que a global, e o R² fica quase zerado mesmo com um
modelo bom.

Este script isola (a): treina e testa **dentro** da mesma família, para as famílias
com n >= MIN_N. Reporta junto a razão DP_intra/DP_global de cada alvo em cada
família, que é a chave para ler o número.

R² por família é calculado sobre as predições out-of-fold empilhadas (uma estimativa
por repetição), não como média de R² de fold — com n pequeno a média de R² de fold é
instável.

Uso:
    SCALE_DATA=<dir> SCALE_EMB20B=<dir> python code/03_analysis/within_family_cv.py [--out DIR]
"""
import os, sys, json, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_analysis import DATA, load_20b, kmer_matrix, gclen, SEED, SPLITS, REPEATS

PRIMARY = [("coding_fraction", False), ("gene_density", False), ("noncoding_bp", True),
           ("n_genes", True), ("mean_intergenic_len", True), ("overlap_bp", True)]
MIN_N = 30
ALPHAS = [1, 10, 100, 1000]


def oof_r2(X, y, n_splits=SPLITS, repeats=REPEATS):
    """R² sobre predições out-of-fold empilhadas, média entre repetições."""
    out = []
    for r in range(repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + r)
        pred = np.empty_like(y, dtype=float)
        for tr, te in cv.split(X):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        out.append(r2_score(y, pred))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "..", "results", "json"))
    a = ap.parse_args()

    X20, acc20 = load_20b(18, "fp8")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")

    md = pd.DataFrame(index=accs)
    md["family"] = man.reindex(accs)["family"].replace("", np.nan).values
    for c, _ in PRIMARY:
        md[c] = fe.reindex(accs)[c].values
    md["GC"] = fe.reindex(accs)["GC"].values
    md["genome_length"] = fe.reindex(accs)["genome_length"].values

    keep = md["family"].notna().values
    accs_k, md = accs[keep], md[keep]
    X20_k = X20.astype(np.float32)[keep]
    fam = md["family"].values.astype(str)

    print(f"[within_family] {len(accs_k)} genomas com família", flush=True)
    print("[within_family] montando 6-mer e GC+len ...", flush=True)
    KM = kmer_matrix(accs_k.tolist())
    GL = gclen(md)
    reps = {"evo2_20b_blocks18": X20_k, "6mer": KM, "gc_len": GL}

    sizes = pd.Series(fam).value_counts()
    fams = sizes[sizes >= MIN_N].index.tolist()
    print(f"[within_family] famílias com n >= {MIN_N}: {len(fams)}", flush=True)

    out = {"config": {"min_n": MIN_N, "splits": SPLITS, "repeats": REPEATS, "seed": SEED,
                      "alphas": ALPHAS, "families": {f: int(sizes[f]) for f in fams}},
           "targets": {}}

    for tname, logt in PRIMARY:
        yall = md[tname].values.astype(float)
        yall = np.log1p(np.clip(yall, 0, None)) if logt else yall
        sd_global = float(np.nanstd(yall))
        rec = {"log1p": logt, "sd_global": sd_global, "families": {}}
        print(f"\n[within_family] === {tname} (DP global={sd_global:.3f}) ===", flush=True)
        print(f"  {'família':<22}{'n':>5}{'DPi/DPg':>9}{'evo2':>9}{'6mer':>9}{'gc_len':>9}")
        for f in fams:
            m = (fam == f) & np.isfinite(yall)
            if m.sum() < MIN_N: continue
            y = yall[m]
            sd_in = float(np.std(y))
            if sd_in == 0: continue
            e = {}
            for rname, R in reps.items():
                mu, sd = oof_r2(R[m], y)
                e[rname] = {"r2": mu, "sd": sd}
            rec["families"][f] = {"n": int(m.sum()), "sd_within": sd_in,
                                  "sd_ratio": sd_in / sd_global if sd_global else float("nan"),
                                  "reps": e}
            print(f"  {f:<22}{m.sum():>5}{sd_in/sd_global:>9.2f}"
                  f"{e['evo2_20b_blocks18']['r2']:>9.3f}{e['6mer']['r2']:>9.3f}"
                  f"{e['gc_len']['r2']:>9.3f}", flush=True)

        # resumo: mediana entre famílias + Wilcoxon pareado por família (exploratório)
        ev = [v["reps"]["evo2_20b_blocks18"]["r2"] for v in rec["families"].values()]
        km = [v["reps"]["6mer"]["r2"] for v in rec["families"].values()]
        gl = [v["reps"]["gc_len"]["r2"] for v in rec["families"].values()]
        summ = {"n_families": len(ev),
                "median_r2": {"evo2_20b_blocks18": float(np.median(ev)),
                              "6mer": float(np.median(km)), "gc_len": float(np.median(gl))},
                "n_families_evo2_positive": int(sum(1 for x in ev if x > 0))}
        if len(ev) >= 5:
            w, p = stats.wilcoxon(ev, km)
            summ["wilcoxon_evo2_vs_6mer"] = {"stat": float(w), "p": float(p)}
            w2, p2 = stats.wilcoxon(ev, gl)
            summ["wilcoxon_evo2_vs_gclen"] = {"stat": float(w2), "p": float(p2)}
        rec["summary"] = summ
        print(f"  -> mediana entre famílias: evo2={summ['median_r2']['evo2_20b_blocks18']:.3f} "
              f"6mer={summ['median_r2']['6mer']:.3f} gc_len={summ['median_r2']['gc_len']:.3f} "
              f"| evo2 R2>0 em {summ['n_families_evo2_positive']}/{len(ev)}", flush=True)
        out["targets"][tname] = rec

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "within_family_cv_metrics.json")
    with open(dst, "w") as fh: json.dump(out, fh, indent=1)
    print(f"\n[within_family] -> {dst}")


if __name__ == "__main__":
    main()
