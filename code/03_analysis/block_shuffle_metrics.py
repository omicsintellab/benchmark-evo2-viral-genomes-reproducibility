#!/usr/bin/env python3
"""block_shuffle_metrics.py — métricas do controle de embaralhamento de blocos (R1 3.3).

O `.npz` produzido por `06_gpu/01_block_shuffle.py` traz, para cada genoma, o embedding do
original e o de cada condição embaralhada. Este script converte isso nas duas grandezas que
respondem ao revisor, e em nenhuma outra.

PERGUNTA 1 — o embedding se move quando a ordem dos blocos muda?
    Deslocamento absoluto não significa nada sozinho: qualquer perturbação move um vetor de
    8.192 dimensões em ALGUMA medida. O número só é interpretável contra uma escala, e a
    escala relevante é a distância entre genomas DIFERENTES — que é a distância que o probe
    de fato usa. Reportamos, portanto, o deslocamento como fração da distância típica entre
    genomas (mediana das distâncias par-a-par entre os originais).

    Leitura: fração pequena = o embedding é praticamente invariante ao rearranjo de longo
    alcance, e a claim tem de recuar para features-resumo de nível genômico. Fração da ordem
    de 1 = rearranjar equivale a trocar de genoma, e há representação de arranjo.

PERGUNTA 2 — a informação decodificável sobrevive ao rearranjo?
    Mais direta e mais forte que a distância: treina-se o probe nos embeddings ORIGINAIS e
    prediz-se o alvo a partir dos embeddings EMBARALHADOS. Se o R² não cair, o que o probe lê
    é invariante à ordem — isto é, o probe não estava usando arranjo. Essa é a forma em que a
    resposta a R1 3.3 é uma medida, e não uma opinião sobre distâncias.

RESSALVA QUE VAI NO TEXTO — declarada aqui porque é limitação de desenho, não de execução:
    o embaralhamento também desloca as janelas de 32 kb em relação à sequência, então parte do
    deslocamento medido é efeito de fronteira de janela, não de rearranjo. Sem uma condição de
    rotação circular (que preserva a ordem e move as janelas do mesmo jeito) não é possível
    separar as duas. O deslocamento medido é, portanto, um LIMITE SUPERIOR do efeito de
    rearranjo — o que só reforça a conclusão se ele for pequeno.

Uso:
    python block_shuffle_metrics.py --npz block_shuffle_evo2_20b_L18_w32768.npz \
        --features genome_features.tsv.gz --out results/json/block_shuffle_metrics.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, wilcoxon

# os seis alvos primários de §4.4-bis do PLANO
REG_FEATS = ["coding_fraction", "gene_density", "noncoding_bp", "n_genes",
             "mean_intergenic_len", "overlap_bp"]
ALPHAS = np.logspace(-2, 4, 13)


def log(m):
    print(m, flush=True)


def cos_dist(A, B):
    """Distância de cosseno linha a linha."""
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.sum(a * b, axis=1)


def between_genome_scale(X, rng, n_pairs=20000):
    """Distância de cosseno típica entre genomas DIFERENTES — a régua do deslocamento."""
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

    # Cada condição pode cobrir um subconjunto de `accs` (genoma curto demais para aquele
    # n_blocks). `<cond>_idx` diz a quais linhas de `original` ela corresponde. Arquivos
    # antigos, sem esse campo, tinham todas as condições alinhadas — tratados como cobertura
    # total para não quebrar a releitura de resultados já publicados.
    rows_of = {c: (d[f"{c}_idx"].astype(int) if f"{c}_idx" in d.files
                   else np.arange(len(accs))) for c in conds}
    log(f"genomas: {len(accs)} | condições: original + {conds}")
    for c in conds:
        n = len(rows_of[c])
        if n < len(accs):
            log(f"  AVISO: {c} cobre {n}/{len(accs)} genomas — os ausentes eram curtos demais "
                f"para {c[4:]} blocos, o que enviesa ESSA condição para genomas longos")

    feats = pd.read_csv(a.features, sep="\t").set_index("accession")
    miss = [x for x in accs if x not in feats.index]
    log(f"CONTROLE POSITIVO — accessions no arquivo de features: "
        f"{len(accs)-len(miss)}/{len(accs)}")
    if miss:
        log(f"  AVISO: {len(miss)} ausentes, excluídas da transferência do probe")

    out = {"n_genomes": len(accs), "conditions": conds,
           # basename apenas: o caminho absoluto carrega o nome do usuario e do host
           # para dentro de um JSON que vai para repositorio publico
           "npz": os.path.basename(a.npz)}

    # ---- 1. deslocamento, contra a régua entre-genomas
    med, p5, p95 = between_genome_scale(X0, rng)
    out["between_genome_cosine"] = {"median": med, "p5": p5, "p95": p95}
    log(f"\nrégua entre genomas: cosseno mediano {med:.4f} (p5 {p5:.4f}, p95 {p95:.4f})")

    out["displacement"] = {}
    log("\n=== DESLOCAMENTO DO EMBEDDING ===")
    log(f"{'condição':>10} {'n':>5} {'cos mediano':>12} {'frac da régua':>14} {'IQR':>18}")
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

    # dose-resposta: o deslocamento cresce com o número de blocos?
    # A dose-resposta compara condições ENTRE SI, então só vale no subconjunto que todas
    # cobrem — misturar populações faria a curva refletir composição de amostra, não dose.
    common = sorted(set.intersection(*[set(rows_of[c].tolist()) for c in conds])) if conds else []
    pos = {c: {int(g): k for k, g in enumerate(rows_of[c])} for c in conds}
    nb = np.array([int(c[4:]) for c in conds], dtype=float)
    per_genome = np.stack([
        cos_dist(X0[common], d[c].astype(np.float64)[[pos[c][g] for g in common]])
        for c in conds]) if common else np.zeros((len(conds), 0))
    log(f"\ndose-resposta calculada no subconjunto comum a todas as condições: "
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

    # ---- 2. transferência do probe: treina no original, prediz no embaralhado
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
                # `te` indexa `idx` (linhas de X0); traduzir para as linhas da condição,
                # deixando NaN onde a condição não cobre o genoma
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
