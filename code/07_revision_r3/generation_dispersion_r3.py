#!/usr/bin/env python3
"""generation_dispersion_r3.py — dispersion and paired tests for Figure 2 (reviewer point #14).

Figure 2B and 2C were drawn from group means only: the cached summary
(results/json/generation_summary_evo2_20b.json) carries no dispersion for the completion
metrics. The per-genome rows of the original run do exist, so the Evo 2 versus Markov
comparison can be made paired, within genome, which is how the figure is read.

Reports, per group (held-out eukaryotic viruses, bacteriophage comparator):
  - mean, SD and 95% CI of each quantity;
  - the paired Evo 2 minus Markov difference with its 95% CI, paired t-test, Wilcoxon
    signed-rank test and Cohen's d, on teacher-forced bits per nucleotide over the true gap
    and on 4-mer cosine similarity to the true gap;
  - the between-group difference in genome cross-entropy (Figure 2A), unpaired (Welch).

These are exploratory analyses under Section 2.8: estimates with intervals, nominal and
uncorrected p-values.

Usage:
    python code/07_revision_r3/generation_dispersion_r3.py

The per-genome CSVs ship in results/, which is the default --csv-dir; pass --csv-dir
(or set GEN_CSV_DIR) to read them from elsewhere.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
GROUPS = ["euk_heldout", "phage_seen"]


def ci_mean(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return [float(x.mean() - t * se), float(x.mean() + t * se)]


def describe(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n": int(len(x)),
            "ci95": ci_mean(x)}


def paired(a, b):
    """a - b, paired within genome. a and b are aligned arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = float("nan"), float("nan")
    return {"delta_mean": float(d.mean()), "ci95": ci_mean(d), "n_pairs": int(len(d)),
            "t_paired": float(t), "p_paired_t": float(p),
            "wilcoxon_p": float(pw), "cohens_d": float(d.mean() / d.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.environ.get("GEN_CSV_DIR",
                                                        os.path.join(REPO, "results")))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "json",
                                                  "generation_dispersion_r3.json"))
    a = ap.parse_args()
    if not a.csv_dir:
        raise SystemExit("[erro] informe --csv-dir (ou GEN_CSV_DIR) com os CSVs por genoma")

    comp = pd.read_csv(os.path.join(a.csv_dir, "generation_completion_evo2_20b.csv"))
    perp = pd.read_csv(os.path.join(a.csv_dir, "generation_perplexity_evo2_20b.csv"))
    print(f"[r3] completion: {len(comp)} genomas | perplexity: {len(perp)} genomas")

    out = {"design": {"source": "per-genome rows of the original generation run "
                                "(generation_completion_evo2_20b.csv, "
                                "generation_perplexity_evo2_20b.csv)",
                      "status": "exploratory (Section 2.8): estimates with intervals, "
                                "nominal uncorrected p-values",
                      "n_per_group": {g: int((comp.set_label == g).sum()) for g in GROUPS}},
           "panel_B_completion_bits": {}, "panel_C_kmer_cosine": {},
           "panel_A_genome_cross_entropy": {}}

    for g in GROUPS:
        s = comp[comp.set_label == g]
        out["panel_B_completion_bits"][g] = {
            "evo2": describe(s.bits_evo), "markov": describe(s.bits_markov),
            "paired_evo2_minus_markov": paired(s.bits_evo, s.bits_markov)}
        out["panel_C_kmer_cosine"][g] = {
            "evo2": describe(s.kmer_evo), "markov": describe(s.kmer_markov),
            "paired_evo2_minus_markov": paired(s.kmer_evo, s.kmer_markov)}

    pcol = "bits_per_nt" if "bits_per_nt" in perp.columns else perp.columns[-1]
    for g in GROUPS:
        out["panel_A_genome_cross_entropy"][g] = describe(perp.loc[perp.set_label == g, pcol])
    xe = [perp.loc[perp.set_label == g, pcol].values for g in GROUPS]
    t, p = stats.ttest_ind(xe[0], xe[1], equal_var=False)
    diff = xe[0].mean() - xe[1].mean()
    se = np.sqrt(xe[0].var(ddof=1) / len(xe[0]) + xe[1].var(ddof=1) / len(xe[1]))
    df = len(xe[0]) + len(xe[1]) - 2
    tc = stats.t.ppf(0.975, df)
    out["panel_A_genome_cross_entropy"]["difference_euk_minus_phage"] = {
        "delta_mean": float(diff), "ci95": [float(diff - tc * se), float(diff + tc * se)],
        "welch_t": float(t), "p": float(p), "note": "unpaired; the adjusted comparison is in "
                                                    "generation_matched_metrics.json"}

    print(f"\n[r3] coluna de cross-entropy usada: {pcol}")
    for panel, key, better in (("B", "panel_B_completion_bits", "lower"),
                               ("C", "panel_C_kmer_cosine", "higher")):
        print(f"\n[r3] painel {panel} ({better} is better), Evo 2 - Markov pareado por genoma:")
        for g in GROUPS:
            r = out[key][g]
            pr = r["paired_evo2_minus_markov"]
            print(f"  {g:>12}: evo2 {r['evo2']['mean']:.3f} "
                  f"[{r['evo2']['ci95'][0]:.3f},{r['evo2']['ci95'][1]:.3f}] | "
                  f"markov {r['markov']['mean']:.3f} "
                  f"[{r['markov']['ci95'][0]:.3f},{r['markov']['ci95'][1]:.3f}] | "
                  f"delta {pr['delta_mean']:+.3f} "
                  f"[{pr['ci95'][0]:+.3f},{pr['ci95'][1]:+.3f}] "
                  f"d={pr['cohens_d']:+.2f} p={pr['p_paired_t']:.2e} "
                  f"(Wilcoxon {pr['wilcoxon_p']:.2e})")
    A = out["panel_A_genome_cross_entropy"]
    print(f"\n[r3] painel A: euk {A['euk_heldout']['mean']:.3f} "
          f"[{A['euk_heldout']['ci95'][0]:.3f},{A['euk_heldout']['ci95'][1]:.3f}] | "
          f"phage {A['phage_seen']['mean']:.3f} "
          f"[{A['phage_seen']['ci95'][0]:.3f},{A['phage_seen']['ci95'][1]:.3f}] | "
          f"diff {A['difference_euk_minus_phage']['delta_mean']:+.3f} "
          f"{A['difference_euk_minus_phage']['ci95']} p={A['difference_euk_minus_phage']['p']:.2e}")

    dst = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[r3] -> {dst}")


if __name__ == "__main__":
    main()
