#!/usr/bin/env python3
"""final_statistics.py — the final statistics of the revision (R1 3.5, R2 #6).

Recomputes no cross-validation: it reads the per-fold scores already cached in
`family_cv_metrics.json` and `composition_baselines_metrics.json` and produces the definitive
statistical block. This matters because the PRIMARY scheme changed after the runs (back to the
pre-registered `cl95`) while the scores of all schemes were already saved; re-running the
cross-validation would be wasted compute and would introduce seed variation.

What the reviewers ask for and what comes out of here:
  - R1 3.5 / R2 #6: a test appropriate for repeated cross-validation (Nadeau-Bengio, which
    inflates the variance by (1/n + n_test/n_train)), with an interval and an effect size
    beside every p-value, and multiple-comparison correction inside a small, declared primary
    set.
  - Primary set: six contrasts, Evo 2 20B blocks.18 versus 6-mer, under `cl95`.
  - Pre-declared negative control (cpg_oe, upa_oe), OUTSIDE the correction.
  - Sensitivity: the same contrasts under family grouping.

Note on intervals: the intervals reported in the paper come from `ci_consistency.py`, which
inverts the same corrected statistic as the test. The bootstrap over per-fold differences was
abandoned because it treats those folds as independent.

Usage:
    python final_statistics.py --out ../../results/json
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from family_cv import PRIMARY, NEGCTRL, nadeau_bengio, boot_ci, holm, SPLITS

ROOT = os.path.join(HERE, "..", "..")


def cohens_d(a, b):
    """Effect size for paired observations (Cohen's d over the differences)."""
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def contrast(e, b, label_a, label_b):
    t, p = nadeau_bengio(e, b, SPLITS)
    lo, hi = boot_ci(e, b)
    return {"a": label_a, "b": label_b,
            "r2_a": float(np.mean(e)), "r2_b": float(np.mean(b)),
            "delta": float(np.mean(e) - np.mean(b)), "ci95": [lo, hi],
            "cohens_d": cohens_d(e, b), "t_nadeau_bengio": t, "p_raw": p,
            "n_folds": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "json"))
    a = ap.parse_args()

    fam = json.load(open(os.path.join(ROOT, "results", "json", "family_cv_metrics.json")))
    comp_path = os.path.join(ROOT, "results", "json", "composition_baselines_metrics.json")
    comp = json.load(open(comp_path)) if os.path.exists(comp_path) else None

    EVO, KMER = "evo2_20b_blocks18", "6mer"
    out = {"design": {
        "primary_scheme": "cl95",
        "primary_scheme_rationale":
            "PRE-REGISTERED scheme (corpus_design.yaml: dedup min_seq_id 0.95, "
            "split cluster_aware). Grouping by family answers a different question "
            "(taxonomic independence) and was not the design fixed before the data; "
            "promoting it to primary would be the same post-hoc selection R1 3.5 criticises.",
        "primary_set": [n for n, _ in PRIMARY],
        "contrast": f"{EVO} vs {KMER}",
        "test": "corrected resampled t-test (Nadeau & Bengio, 2003), factor (1/n + 1/(k-1))",
        "correction": "Holm within the primary set of six",
        "ci": "obtained by inverting the SAME corrected statistic as the test "
               "(see ci_consistency.json); the bootstrap over per-fold differences treats "
               "those folds as independent and was abandoned",
        "negative_control": [n for n, _ in NEGCTRL],
        "negative_control_note":
            "Dinucleotide-composition targets, declared IN ADVANCE as cases where "
            "the 6-mer baseline is expected to win. These are not hypothesis tests but a "
            "a specificity check on the probe. OUTSIDE the correction.",
        "exploratory":
            "GC-and-length baseline; Baltimore/host/family classification; layer "
            "sweep; 20B versus 7B; precision control; PCA control; random CV; "
            "the whole generative evaluation. Reported with intervals, without confirmatory p-values.",
    }, "primary": {}, "sensitivity_family": {}, "negative_control": {}}

    # ---- primary: cl95
    pv, names = [], []
    for name, _ in PRIMARY:
        r = fam["targets"][name]["reps"]
        c = contrast(r[EVO]["cl95"]["scores"], r[KMER]["cl95"]["scores"], EVO, KMER)
        out["primary"][name] = c
        pv.append(c["p_raw"]); names.append(name)
    adj = holm(pv, names)
    for name in names:
        out["primary"][name]["p_holm"] = adj[name]
        out["primary"][name]["significant_holm"] = bool(adj[name] < 0.05 and
                                                        out["primary"][name]["delta"] > 0)

    # ---- sensitivity: the same contrasts under family grouping
    pv2 = []
    for name, _ in PRIMARY:
        r = fam["targets"][name]["reps"]
        c = contrast(r[EVO]["family"]["scores"], r[KMER]["family"]["scores"], EVO, KMER)
        out["sensitivity_family"][name] = c
        pv2.append(c["p_raw"])
    adj2 = holm(pv2, names)
    for name in names:
        out["sensitivity_family"][name]["p_holm"] = adj2[name]

    # ---- negative control, no correction
    for name, _ in NEGCTRL:
        r = fam["targets"][name]["reps"]
        out["negative_control"][name] = {
            "cl95": contrast(r[EVO]["cl95"]["scores"], r[KMER]["cl95"]["scores"], EVO, KMER),
            "family": contrast(r[EVO]["family"]["scores"], r[KMER]["family"]["scores"],
                               EVO, KMER)}

    # ---- strongest baseline per class, when available
    if comp:
        out["vs_best_baseline_by_class"] = {
            sch: {t: {cls: {k: v[k] for k in ("best", "best_r2", "delta", "ci95", "p",
                                              "evo2_ahead")}
                      for cls, v in tv["by_class"].items()}
                  for t, tv in comp["tests"][sch].items()}
            for sch in ("cl95", "family")}
        out["design"]["baseline_classes"] = comp["classes"]

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "final_statistics.json")
    json.dump(out, open(dst, "w"), indent=1)

    def row(n, c, pkey):
        ci = f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
        print(f"{n:<22}{c['r2_a']:>8.3f}{c['r2_b']:>8.3f}{c['delta']:>8.3f}{ci:>16}"
              f"{c['cohens_d']:>7.2f}{c[pkey]:>10.2g}")

    hdr = f"\n{'alvo':<22}{'Evo2':>8}{'6-mer':>8}{'dR2':>8}{'IC95':>16}{'d':>7}{'p':>10}"
    print(hdr)
    print("-- PRIMARIO (cl95, pre-registrado), p corrigido por Holm " + "-" * 18)
    for n in names:
        row(n, out["primary"][n], "p_holm")
    print("-- SENSITIVITY (family-grouped), Holm-corrected p " + "-" * 22)
    for n in names:
        row(n, out["sensitivity_family"][n], "p_holm")
    print("-- CONTROLE NEGATIVO (fora da correcao), p bruto " + "-" * 26)
    for n, _ in NEGCTRL:
        row(n, out["negative_control"][n]["cl95"], "p_raw")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
