#!/usr/bin/env python3
"""ci_consistency.py — make the confidence interval consistent with the test (R1 3.5, R2 #6).

`final_statistics.py` reported, side by side, a p-value corrected by Nadeau-Bengio and a
confidence interval bootstrapped over the per-fold differences. The two rest on different
variances:

  - Nadeau-Bengio inflates the variance by (1/n + 1/(k-1)), because folds of repeated
    cross-validation share training data and are NOT independent;
  - the bootstrap over per-fold differences treats those folds as independent, which is
    exactly the assumption R1 3.5 and R2 #6 reject.

With k = 5 and n = 15 the factor is 1/15 + 1/4 = 0.3167 against 1/15 = 0.0667, so the honest
standard error is 2.18x larger. The symptom shows under family grouping, where primary targets
have a bootstrap interval that EXCLUDES zero next to p = 0.11-0.22.

The interval consistent with the test is obtained by inverting the statistic itself:

    CI = d_bar +/- t_{0.975, n-1} * sqrt((1/n + 1/(k-1)) * s^2_d)

No value is typed by hand: everything recomputes from the per-fold scores already cached in
`family_cv_metrics.json`, without re-running any cross-validation (which would introduce seed
variation).

Usage:
    python ci_consistency.py --json ../../results/json --out ../../results/json/ci_consistency.json
"""
import argparse, json, os
import numpy as np
from scipy import stats

PRIMARY = ["coding_fraction", "gene_density", "noncoding_bp", "n_genes",
           "mean_intergenic_len", "overlap_bp"]
NEGCTRL = ["cpg_oe", "upa_oe"]
EVO = "evo2_20b_blocks18"
BASE = "6mer"
SPLITS = 5          # folds per repeat (k)
SEED = 42


def log(m):
    print(m, flush=True)


def nb_stats(e, b, k=SPLITS):
    """Devolve (delta, t, p, IC_NB, IC_bootstrap, n_folds) para duas listas de scores."""
    d = np.asarray(e, float) - np.asarray(b, float)
    n = len(d)
    if n < 2:
        return None
    var = d.var(ddof=1)
    corr = 1.0 / n + 1.0 / (k - 1)
    se_nb = np.sqrt(corr * var)
    t = d.mean() / se_nb if se_nb > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), n - 1)) if np.isfinite(t) else np.nan
    tcrit = stats.t.ppf(0.975, n - 1)
    ci_nb = [float(d.mean() - tcrit * se_nb), float(d.mean() + tcrit * se_nb)]

    rng = np.random.default_rng(SEED)
    bs = rng.choice(d, size=(10000, n), replace=True).mean(axis=1)
    ci_bs = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    return {"delta": float(d.mean()), "cohens_d": float(d.mean() / d.std(ddof=1))
            if d.std(ddof=1) > 0 else float("nan"),
            "t_nadeau_bengio": float(t), "p_raw": p,
            "ci95_nb": ci_nb, "ci95_bootstrap": ci_bs, "n_folds": int(n),
            "se_ratio_nb_over_bootstrap": float(np.sqrt(corr / (1.0 / n)))}


def holm(pvals, names):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return {names[i]: adj[i] for i in range(m)}


def excludes_zero(ci):
    return (ci[0] > 0) or (ci[1] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../../results/json")
    ap.add_argument("--out", default="../../results/json/ci_consistency.json")
    a = ap.parse_args()

    fam = json.load(open(os.path.join(a.json, "family_cv_metrics.json")))
    tg = fam["targets"]

    out = {"note": ("Interval recomputed with the SAME variance as the Nadeau-Bengio test. "
                    "A bootstrap interval over per-fold differences is anti-conservative because "
                    "trata folds de CV repetida como independentes."),
           "correction_factor": f"1/n + 1/(k-1), k={SPLITS}",
           "schemes": {}}

    for scheme in ("cl95", "family"):
        res, ps, names = {}, [], []
        for t in PRIMARY:
            if t not in tg:
                continue
            reps = tg[t]["reps"]
            if EVO not in reps or BASE not in reps:
                continue
            if scheme not in reps[EVO] or scheme not in reps[BASE]:
                continue
            r = nb_stats(reps[EVO][scheme]["scores"], reps[BASE][scheme]["scores"])
            if r is None:
                continue
            res[t] = r; ps.append(r["p_raw"]); names.append(t)
        adj = holm(ps, names)
        for t in names:
            res[t]["p_holm"] = adj[t]
            res[t]["significant_holm"] = bool(adj[t] < 0.05)
            res[t]["ci_nb_excludes_zero"] = excludes_zero(res[t]["ci95_nb"])
            res[t]["ci_bootstrap_excludes_zero"] = excludes_zero(res[t]["ci95_bootstrap"])
            res[t]["disagreement"] = bool(
                res[t]["ci_bootstrap_excludes_zero"] != res[t]["significant_holm"])
        out["schemes"][scheme] = res

    # ---- contrastes contra o MELHOR baseline de cada classe
    # This is where the bootstrap-vs-test conflict actually shows: under family grouping the
    # best compositional baseline becomes GC-and-length, and the deltas get small enough that
    # the choice of variance decides the verdict.
    comp = json.load(open(os.path.join(a.json, "composition_baselines_metrics.json")))
    CLASSES = {"compositional": ["kmer3", "kmer4", "kmer5", "kmer6", "multik",
                                 "codon", "dicodon", "gc_len"],
               "annotation_derived": ["orf", "compo_all"]}
    out["vs_best_baseline_by_class"] = {}
    for scheme in ("cl95", "family"):
        out["vs_best_baseline_by_class"][scheme] = {}
        for t in PRIMARY:
            if t not in comp["targets"]:
                continue
            node = comp["targets"][t]["reps"]
            if EVO not in node or scheme not in node[EVO]:
                continue
            evo = node[EVO][scheme]["scores"]
            out["vs_best_baseline_by_class"][scheme][t] = {}
            for cls, members in CLASSES.items():
                av = [m for m in members if m in node and scheme in node[m]]
                if not av:
                    continue
                best = max(av, key=lambda m: node[m][scheme]["mean"])
                r = nb_stats(evo, node[best][scheme]["scores"])
                if r is None:
                    continue
                r["best"] = best
                r["best_r2"] = float(node[best][scheme]["mean"])
                r["evo2_ahead"] = bool(r["p_raw"] < 0.05 and r["delta"] > 0)
                r["evo2_behind"] = bool(r["p_raw"] < 0.05 and r["delta"] < 0)
                r["ci_nb_excludes_zero"] = excludes_zero(r["ci95_nb"])
                r["ci_bootstrap_excludes_zero"] = excludes_zero(r["ci95_bootstrap"])
                r["disagreement"] = bool(
                    r["ci_bootstrap_excludes_zero"] != (r["p_raw"] < 0.05))
                out["vs_best_baseline_by_class"][scheme][t][cls] = r

    # ---- relato
    for scheme, res in out["schemes"].items():
        log(f"\n=== ESQUEMA {scheme} — Evo 2 20B blocks.18 vs 6-mer ===")
        log(f"{'alvo':>22} {'ΔR²':>7} {'IC bootstrap':>20} {'IC Nadeau-Bengio':>22} "
            f"{'p (Holm)':>10} {'conflito':>9}")
        for t, r in res.items():
            cb = f"[{r['ci95_bootstrap'][0]:+.3f}; {r['ci95_bootstrap'][1]:+.3f}]"
            cn = f"[{r['ci95_nb'][0]:+.3f}; {r['ci95_nb'][1]:+.3f}]"
            log(f"{t:>22} {r['delta']:+7.3f} {cb:>20} {cn:>22} {r['p_holm']:10.3g} "
                f"{'SIM' if r['disagreement'] else '-':>9}")
        nd = sum(1 for r in res.values() if r["disagreement"])
        log(f"  -> {nd} alvo(s) em que o IC bootstrap contradiz o teste")
        if res:
            log(f"  -> honest standard error is {list(res.values())[0]['se_ratio_nb_over_bootstrap']:.2f}x "
                f"o do bootstrap")

    log("\n=== CONTRA O MELHOR BASELINE DE CADA CLASSE ===")
    log(f"{'esquema':>7} {'alvo':>21} {'classe':>19} {'melhor':>8} {'ΔR²':>7} "
        f"{'IC bootstrap':>19} {'IC Nadeau-Bengio':>19} {'p':>9} {'conflito':>8}")
    ndis = 0
    for scheme, per_t in out["vs_best_baseline_by_class"].items():
        for t, per_c in per_t.items():
            for cls, r in per_c.items():
                ndis += r["disagreement"]
                cb = f"[{r['ci95_bootstrap'][0]:+.3f};{r['ci95_bootstrap'][1]:+.3f}]"
                cn = f"[{r['ci95_nb'][0]:+.3f};{r['ci95_nb'][1]:+.3f}]"
                log(f"{scheme:>7} {t:>21} {cls:>19} {r['best']:>8} {r['delta']:+7.3f} "
                    f"{cb:>19} {cn:>19} {r['p_raw']:9.3g} "
                    f"{'SIM' if r['disagreement'] else '-':>8}")
    out["n_disagreements_by_class"] = int(ndis)
    log(f"\n  -> {ndis} contraste(s) em que o IC bootstrap contradiz o teste; "
        f"com o IC de Nadeau-Bengio a contradicao desaparece em todos")

    # placar por classe, para o texto — contado, nunca digitado
    out["scoreboard"] = {}
    for scheme, per_t in out["vs_best_baseline_by_class"].items():
        out["scoreboard"][scheme] = {}
        for cls in ("compositional", "annotation_derived"):
            ahead = [t for t, pc in per_t.items() if pc.get(cls, {}).get("evo2_ahead")]
            behind = [t for t, pc in per_t.items() if pc.get(cls, {}).get("evo2_behind")]
            out["scoreboard"][scheme][cls] = {
                "n_targets": len(per_t), "evo2_ahead": ahead, "evo2_behind": behind}
            log(f"  {scheme:>7} {cls:>19}: a frente em {len(ahead)}/{len(per_t)} "
                f"{ahead} | atras em {len(behind)} {behind}")

    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
