#!/usr/bin/env python3
"""make_supp_table_selection.py — tabela de seleção de amostras (R1 4.3, R2 #5).

Os revisores apontaram que 981, 1.080 e 1.912 parecem inconsistentes. Estão certos —
os três números estão corretos, o artigo é que nunca explicou como se compõem. Esta tabela
é a explicação, e **todos os números são contados dos arquivos de dados**, não digitados.
(Substitui o fluxograma que ocupava esse papel; um dos revisores pediu "sample-selection
table or flow diagram", e a tabela cabe melhor no suplementar.)

ATENÇÃO a uma pegadinha dos dados: existem DUAS colunas chamadas `host`.
  - em `corpus_manifest.tsv.gz` é o **domínio** do hospedeiro (eukaryote/bacteria/archaea/
    unknown) — é essa que define a população de 1.080;
  - em `genome_features.tsv.gz` é a **string livre** do organismo hospedeiro ("Homo sapiens",
    "tomato", …), com 282 nulos no mesmo subconjunto.
Usar a segunda no lugar da primeira dá 918 em vez de 1.080 e a tabela passa a mentir.

Uso:
    python code/05_figures/make_supp_table_selection.py --data <dir do repo público>/data
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

    # organismos multi-record (R2 #4), contados na união
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
         "GroupKFold by family and genus, leave-one-family-out, within-family CV"),
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
