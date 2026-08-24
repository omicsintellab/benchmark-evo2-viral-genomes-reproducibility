#!/usr/bin/env python3
"""make_supp_table_selection.py — sample-selection table (R1 4.3, R2 #5).

The reviewers noted that 981, 1,080 and 1,912 look inconsistent. They are right to ask: the
three numbers are correct, but the paper never explained how they compose. This table is that
explanation, and **every number is counted from the data files**, not typed.

Watch out for one trap in the data: there are TWO columns called `host`.
  - in `corpus_manifest.tsv.gz` it is the host **domain** (eukaryote/bacteria/archaea/unknown),
    and that is the one defining the 1,080-record population;
  - in `genome_features.tsv.gz` it is the **free-text** host organism ("Homo sapiens", ...).
Using the second instead of the first yields 918 rather than 1,080, and the table starts lying.

Usage:
    python code/05_figures/make_supp_table_selection.py --data <repo>/data
"""
import argparse, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
DST = os.path.join(REPO, "results", "tables", "supp_S8_selection.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser(
        "~/Documents/GitHub/benchmark-evo2-viral-genomes-reproducibility/data"))
    ap.add_argument("--out", default=DST)
    a = ap.parse_args()

    g = pd.read_csv(os.path.join(a.data, "genome_features.tsv.gz"), sep="\t")
    m = pd.read_csv(os.path.join(a.data, "corpus_manifest.tsv.gz"), sep="\t")
    b = pd.read_csv(os.path.join(a.data, "probe_subset_baltimore.tsv"), sep="\t")
    f = pd.read_csv(os.path.join(a.data, "probe_subset_features.tsv"), sep="\t")

    n_corpus = len(g)
    n_b, n_f = len(b), len(f)
    union = set(b.accession) | set(f.accession)
    n_u, n_i = len(union), len(set(b.accession) & set(f.accession))

    host = m[m.accession.isin(f.accession)]["host"].fillna("unknown")
    n_unknown = int((host == "unknown").sum())
    n_host = n_f - n_unknown
    host_parts = ", ".join(f"{v} {k}" for k, v in host.value_counts().items()
                           if k != "unknown")

    fam = g[g.accession.isin(union)]["family"]
    n_fam, n_nofam = int(fam.notna().sum()), int(fam.isna().sum())

    bal = b["baltimore"].value_counts().sort_index()
    bal_parts = ", ".join(f"{k} {v}" for k, v in bal.items())
    short = [k for k, v in bal.items() if v < 150]

    # multi-record organisms (R2 #4), counted within the union
    org = g[g.accession.isin(union)].groupby("organism").size()
    n_multi_org = int((org > 1).sum())
    n_multi_rec = int(org[org > 1].sum())

    rows = [
        ("Pre-registered RefSeq viral corpus", f"{n_corpus:,}", "—",
         "quota sampling by Baltimore class and host domain"),
        ("Baltimore probe subset", f"{n_b:,}", "—",
         f"{bal_parts}; classes {' and '.join(short)} have fewer than 150 records "
         f"meeting the quota"),
        ("Feature probe subset", f"{n_f:,}", "—",
         "regression targets for genome architecture"),
        ("Host-domain classification", f"{n_host:,}", f"−{n_unknown} unknown host domain",
         host_parts),
        ("Union of both subsets", f"{n_u:,}", "—",
         f"layer sweep, scale and precision controls; intersection of the two subsets = {n_i}"),
        ("Family-grouped analyses", f"{n_fam:,}", f"−{n_nofam} without an assigned family",
         "GroupKFold by family, leave-one-family-out, within-family CV"),
    ]

    out = []
    out.append("## Supplementary Table S8. Sample selection\n")
    out.append("Composition of every analysed population. Counts are derived from the released "
               "data files, not transcribed. This table replaces the flow diagram requested in "
               "review and resolves the apparent inconsistency between 981, 1,080 and 1,912.\n")
    out.append("| Stage | n | Excluded at this step | Composition and use |")
    out.append("|---|---|---|---|")
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    out.append("")
    out.append(f"**Segmented viruses.** Features were extracted one row per record, so a "
               f"segmented virus contributes one row per segment: within the union, "
               f"{n_multi_org} organisms contribute more than one record, totalling "
               f"{n_multi_rec} records ({100 * n_multi_rec / n_u:.1f}%). In the grouped "
               f"analyses family subsumes organism, so all records of one organism fall in the "
               f"same fold; the analysis script aborts if that invariant is violated.\n")
    out.append(f"**Records without a family assignment** ({n_nofam}) are excluded from the "
               "grouped analyses and reported as a sensitivity analysis treating each as its "
               "own group.\n")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
