#!/usr/bin/env python3
"""overlap_sensitivity.py — does the `overlap_bp` contrast survive the corrected definition? (R2 #17)

`overlap_bp` is one of the six primary targets, and re-extraction showed that the published
definition (sum of spans minus union) differs from the correct one (positions covered by two or
more CDS features) in 7.9% of the analysed records, with about 26% of true overlap occurring
between OPPOSITE strands. Before reporting the target, we need to know whether the conclusion
depends on the definition.

Runs the same contrast (Evo 2 20B blocks.18 versus 6-mer), on the same folds, under three
definitions:
  published   — as in the paper
  positions   — positions with coverage >= 2 (the correct one)
  same_strand — same-strand overlap only (the biologically stricter reading)

Usage:
    PYTHONPATH=<repo>/code/03_analysis python overlap_sensitivity.py \
        --provenance ../../results/tables/annotation_provenance.tsv.gz --out ../../results/json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scale_analysis import DATA, load_20b, kmer_matrix, groups as cl95_groups
from family_cv import reg_group, nadeau_bengio, boot_ci, SPLITS

DEFS = {"published": "overlap_bp_published",
        "positions": "overlap_positions",
        "same_strand": "overlap_same_strand"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "json"))
    a = ap.parse_args()

    X20, acc20 = load_20b(18, "fp8")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    prov = pd.read_csv(a.provenance, sep="\t").set_index("accession")
    print(f"CONTROLE POSITIVO — {len(accs)} accessions, "
          f"{prov.reindex(accs)[DEFS['positions']].notna().sum()} with provenance")

    fam = man.reindex(accs)["family"].replace("", np.nan)
    keep = fam.notna().values & prov.reindex(accs)[DEFS["positions"]].notna().values
    accs_k = accs[keep]
    X = X20.astype(np.float32)[keep]
    KM = kmer_matrix(list(accs_k))
    g95 = cl95_groups(accs_k)
    famv = fam[keep].values.astype(str)
    P = prov.reindex(accs_k)
    print(f"population: {len(accs_k)}")

    out = {"n": int(len(accs_k)), "definitions": DEFS, "results": {}}
    print(f"\n{'definicao':<14}{'esquema':>8}{'Evo2':>9}{'6-mer':>9}{'dR2':>9}{'p':>11}")
    for dname, col in DEFS.items():
        y = np.log1p(np.clip(P[col].values.astype(float), 0, None))
        rec = {}
        for sch, g in (("cl95", g95), ("family", famv)):
            e = reg_group(X, y, g)
            k = reg_group(KM, y, g)
            t, p = nadeau_bengio(e, k, SPLITS)
            lo, hi = boot_ci(e, k)
            rec[sch] = {"r2_evo2": float(e.mean()), "r2_6mer": float(k.mean()),
                        "delta": float(e.mean() - k.mean()), "ci95": [lo, hi],
                        "t": t, "p": p}
            print(f"{dname:<14}{sch:>8}{e.mean():>9.3f}{k.mean():>9.3f}"
                  f"{e.mean()-k.mean():>9.3f}{p:>11.2g}")
        out["results"][dname] = rec

    # does the conclusion change?
    base = out["results"]["published"]
    out["conclusion_robust"] = {
        sch: bool(all(out["results"][d][sch]["delta"] > 0 and
                      out["results"][d][sch]["p"] < 0.05 for d in DEFS))
        for sch in ("cl95", "family")}
    print(f"\nconclusao robusta a definicao: {out['conclusion_robust']}")

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "overlap_sensitivity.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
