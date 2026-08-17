#!/usr/bin/env python3
"""generation_matched.py — euk held-out vs fago, ajustado e pareado (R2 #11).

R2 #11: "a comparação generativa não isola o efeito da exposição ao pré-treino. Fagos e
vírus de eucariotos diferem em tipo de genoma, comprimento, GC, densidade codificante,
segmentação e diversidade taxonômica. A diferença em bits por nucleotídeo não deveria ser
atribuída apenas à inclusão no treino. Uma análise pareada ou ajustada seria mais
convincente, e o termo 'seen bacteriophages' deve ser evitado a menos que a sobreposição
exata com o conjunto de treino tenha sido verificada."

Três análises, da mais ingênua à mais defensável:
  1) bruta          — diferença de médias, que é o que o artigo reporta hoje
  2) ajustada       — OLS com log(comprimento), GC, fração codificante e classe de Baltimore
  3) pareada        — vizinho mais próximo em log(comprimento) e GC, sem reposição

Se a diferença sobreviver a (2) e (3), ela não é explicada pelas covariáveis que o revisor
lista. Se encolher muito, o artigo tem de dizer isso — que é o desfecho que R2 #11 antecipa.

**Sobre nomenclatura:** não temos verificação de sobreposição exata com o corpus de treino do
Evo 2. O script emite o alerta e o texto deve trocar "seen bacteriophages" por algo como
"bacteriophages, a group represented in the pre-training corpus".

Uso:
    python generation_matched.py --perp <csv> --comp <csv> --out ../../results/json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scale_analysis import DATA

COVARS = ["log_len", "GC", "coding_fraction"]


def load(perp, comp):
    p = pd.read_csv(perp)
    c = pd.read_csv(comp)
    fe = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")
    df = p.merge(c[["accession", "bits_evo", "bits_markov", "kmer_evo", "kmer_markov"]],
                 on="accession", how="left")
    df["genome_length"] = fe.reindex(df.accession)["genome_length"].values
    df["GC"] = fe.reindex(df.accession)["GC"].values
    df["coding_fraction"] = fe.reindex(df.accession)["coding_fraction"].values
    df["log_len"] = np.log10(df["genome_length"].astype(float))
    df["is_phage"] = (df["set_label"] == "phage_seen").astype(int)
    return df.dropna(subset=["bits_nt", "log_len", "GC", "coding_fraction"])


def describe_imbalance(df):
    """Quão diferentes são os dois grupos NAS COVARIÁVEIS — é o cerne da objeção."""
    out = {}
    for c in COVARS + ["genome_length"]:
        a = df.loc[df.is_phage == 0, c].astype(float)
        b = df.loc[df.is_phage == 1, c].astype(float)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out[c] = {"euk_mean": float(a.mean()), "phage_mean": float(b.mean()),
                  "std_diff": float((b.mean() - a.mean()) / pooled) if pooled else float("nan")}
    return out


def adjusted(df, y):
    import statsmodels.formula.api as smf
    d = df.rename(columns={y: "_y"})
    m0 = smf.ols("_y ~ is_phage", data=d).fit()
    m1 = smf.ols("_y ~ is_phage + log_len + GC + coding_fraction + C(baltimore)",
                 data=d).fit()
    return ({"coef": float(m0.params["is_phage"]), "ci95": [float(x) for x in
             m0.conf_int().loc["is_phage"]], "p": float(m0.pvalues["is_phage"]),
             "n": int(m0.nobs)},
            {"coef": float(m1.params["is_phage"]), "ci95": [float(x) for x in
             m1.conf_int().loc["is_phage"]], "p": float(m1.pvalues["is_phage"]),
             "n": int(m1.nobs), "r2": float(m1.rsquared),
             "covariates": "log_len + GC + coding_fraction + C(baltimore)"})


def matched(df, y, seed=42):
    """Pareamento 1:1 por vizinho mais próximo em (log_len, GC) padronizados, sem reposição."""
    from scipy.spatial import cKDTree
    from scipy import stats
    a = df[df.is_phage == 0].copy()
    b = df[df.is_phage == 1].copy()
    Z = df[["log_len", "GC"]].astype(float)
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    A = ((a[["log_len", "GC"]] - mu) / sd).values
    B = ((b[["log_len", "GC"]] - mu) / sd).values
    tree = cKDTree(B)
    used, pairs = set(), []
    d, idx = tree.query(A, k=min(len(B), 10))
    d, idx = np.atleast_2d(d), np.atleast_2d(idx)
    for i in np.argsort(d[:, 0]):                    # os melhores pares primeiro
        for j, dist in zip(idx[i], d[i]):
            if j not in used:
                used.add(j); pairs.append((i, j, float(dist))); break
    if not pairs:
        return {"n_pairs": 0}
    ya = a.iloc[[p[0] for p in pairs]][y].astype(float).values
    yb = b.iloc[[p[1] for p in pairs]][y].astype(float).values
    diff = yb - ya
    t, p = stats.ttest_rel(yb, ya)
    return {"n_pairs": len(pairs),
            "mean_dist_std_units": float(np.mean([p[2] for p in pairs])),
            "diff_mean": float(diff.mean()),
            "ci95": [float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                     float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))],
            "t": float(t), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perp", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "json"))
    a = ap.parse_args()

    df = load(a.perp, a.comp)
    print(f"[generation_matched] n={len(df)} | "
          f"euk={int((df.is_phage==0).sum())} fago={int((df.is_phage==1).sum())}")

    out = {"n": int(len(df)),
           "naming_warning":
               "Nao ha verificacao de sobreposicao exata com o corpus de treino do Evo 2. "
               "Evitar 'seen bacteriophages'; usar formulacao como 'a group represented in "
               "the pre-training corpus'. (R2 #11)",
           "covariate_imbalance": describe_imbalance(df), "outcomes": {}}

    for y in ["bits_nt", "bits_evo", "kmer_evo"]:
        if y not in df or df[y].isna().all():
            continue
        crude, adj = adjusted(df, y)
        out["outcomes"][y] = {"crude": crude, "adjusted": adj, "matched": matched(df, y)}
        print(f"\n  {y}: bruto={crude['coef']:+.3f} (p={crude['p']:.2g}) | "
              f"ajustado={adj['coef']:+.3f} (p={adj['p']:.2g}) | "
              f"pareado={out['outcomes'][y]['matched']['diff_mean']:+.3f} "
              f"(p={out['outcomes'][y]['matched']['p']:.2g}, "
              f"n={out['outcomes'][y]['matched']['n_pairs']} pares)")

    print("\n  desequilíbrio de covariáveis (diferença padronizada; |d|>0,25 = relevante):")
    for c, v in out["covariate_imbalance"].items():
        print(f"    {c:<18} euk={v['euk_mean']:>10.3f} fago={v['phage_mean']:>10.3f} "
              f"d={v['std_diff']:+.2f}")

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "generation_matched_metrics.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
