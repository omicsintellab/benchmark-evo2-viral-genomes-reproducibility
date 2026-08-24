#!/usr/bin/env python3
"""make_supp_tables_r1.py — the supplementary tables added in revision R1.

Reads cached JSON only; no value is typed by hand. Output goes to results/tables/.

  S5 — family-grouped CV: the six primary contrasts under both schemes (R1 3.2, R2 #1/#2)
  S6 — Evo 2 versus the strongest baseline of each class (R1 3.4, R2 #3)
  S7 — classification beyond accuracy: macro-F1 and balanced accuracy (R2 #15)

Intervals come from inverting the Nadeau-Bengio statistic, not from the bootstrap: see
ci_consistency.py.

Usage:
    python code/05_figures/make_supp_tables_r1.py
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
J = os.path.join(REPO, "results", "json")
DST = os.path.join(REPO, "results", "tables", "supp_S5_S7.md")

NICE = {"coding_fraction": "Coding fraction", "gene_density": "Gene density",
        "noncoding_bp": "Non-coding bp", "n_genes": "Gene count",
        "mean_intergenic_len": "Mean intergenic length", "overlap_bp": "Gene overlap"}
SCHEME = {"cl95": "Identity-clustered (pre-registered)", "family": "Family-grouped"}
CLS = {"compositional": "Compositional", "annotation_derived": "Annotation-derived"}


def fp(p):
    if p != p:
        return "—"
    return "<1e-4" if p < 1e-4 else f"{p:.2g}"


def main():
    cc = json.load(open(os.path.join(J, "ci_consistency.json")))
    cm = json.load(open(os.path.join(J, "classification_metrics.json")))
    out = []

    out.append("## Supplementary Table S5. Primary contrasts under both cross-validation "
               "schemes\n")
    out.append("Evo 2 20B `blocks.18` versus the 6-mer baseline on the six pre-declared "
               "architecture targets. ΔR² is the difference in cross-validated R²; the test is "
               "a Nadeau–Bengio corrected resampled t-test and the interval is obtained by "
               "inverting the same statistic. p-values are Holm-corrected within the set of "
               "six.\n")
    out.append("| Scheme | Target | ΔR² | 95% CI | Cohen's d | p (Holm) |")
    out.append("|---|---|---|---|---|---|")
    for sch in ("cl95", "family"):
        for t, r in cc["schemes"][sch].items():
            out.append(f"| {SCHEME[sch]} | {NICE.get(t, t)} | {r['delta']:+.3f} | "
                       f"[{r['ci95_nb'][0]:+.3f}; {r['ci95_nb'][1]:+.3f}] | "
                       f"{r['cohens_d']:.1f} | {fp(r['p_holm'])} |")
    out.append("")

    out.append("## Supplementary Table S6. Evo 2 versus the best baseline of each class\n")
    out.append("For each target and scheme, the strongest baseline within each class was "
               "selected and contrasted with the embedding. Compositional: k-mer (k = 3–6), "
               "multi-k, codon, dicodon, GC-and-length. Annotation-derived: six-frame ORF scan "
               "and the combined representation containing it. A positive ΔR² favours Evo 2.\n")
    out.append("| Scheme | Target | Baseline class | Best baseline | Baseline R² | ΔR² | "
               "95% CI | p |")
    out.append("|---|---|---|---|---|---|---|---|")
    for sch in ("cl95", "family"):
        for t, per_c in cc["vs_best_baseline_by_class"][sch].items():
            for cls, r in per_c.items():
                out.append(f"| {SCHEME[sch]} | {NICE.get(t, t)} | {CLS[cls]} | "
                           f"`{r['best']}` | {r['best_r2']:.3f} | {r['delta']:+.3f} | "
                           f"[{r['ci95_nb'][0]:+.3f}; {r['ci95_nb'][1]:+.3f}] | "
                           f"{fp(r['p_raw'])} |")
    out.append("")
    sb = cc["scoreboard"]
    out.append("**Summary.** Under the pre-registered scheme Evo 2 leads the compositional "
               f"class on {len(sb['cl95']['compositional']['evo2_ahead'])} of 6 targets and the "
               f"annotation-derived class on {len(sb['cl95']['annotation_derived']['evo2_ahead'])} "
               "of 6. Under family-grouped cross-validation it leads the compositional class on "
               f"{len(sb['family']['compositional']['evo2_ahead'])} of 6 and the "
               "annotation-derived class on "
               f"{len(sb['family']['annotation_derived']['evo2_ahead'])} of 6, and is "
               "significantly behind the ORF baseline on "
               f"{', '.join(NICE.get(x, x).lower() for x in sb['family']['annotation_derived']['evo2_behind']) or 'no target'}.\n")

    out.append("## Supplementary Table S7. Classification beyond accuracy\n")
    out.append("Under the pre-registered identity-clustered scheme. Balanced accuracy and "
               "macro-F1 are reported because the classes are unbalanced (R2 #15).\n")
    rows = _classification_rows(cm)
    out.append("| Target | n | Classes | Representation | Accuracy | Balanced accuracy | "
               "Macro-F1 |")
    out.append("|---|---|---|---|---|---|---|")
    out.extend(rows)
    out.append("")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    open(DST, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n-> {DST}")


def _classification_rows(cm, scheme="cl95"):
    """Extrai as linhas de S7. Estrutura: targets[alvo].reps[rep][esquema]."""
    rows = []
    for tgt, d in cm.get("targets", {}).items():
        n = d.get("n", "—")
        for rep, per_scheme in d.get("reps", {}).items():
            m = per_scheme.get(scheme)
            if not isinstance(m, dict) or "accuracy" not in m:
                continue
            rows.append(
                f"| {tgt} | {n} | {m.get('n_classes', '—')} | `{rep}` | "
                f"{m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | "
                f"{m['macro_f1']:.3f} |")
    return rows


if __name__ == "__main__":
    main()
