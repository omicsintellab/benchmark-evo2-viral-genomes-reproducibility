#!/usr/bin/env python3
"""reextract_annotation.py — annotation provenance and corrected overlap (R1 4.1, R2 #16/#17).

R2 #16 notes that substituting CDS counts for gene counts when `gene` features are absent
creates a mixed definition, and asks how often that fallback was used, plus a sensitivity
analysis restricted to consistently annotated genomes.

R2 #17 notes that "summed CDS length minus union length" is not always identical to the number
of positions covered by more than one CDS, and asks for a precise description of the
implemented calculation and of the treatment of opposite strands and multiple overlaps.

Both are correct. The published `extract_genome_features.py` computes
`overlap_bp = sum(spans) - union` over spans collected in the `t == "CDS"` branch, which:
  - DOUBLE COUNTS positions where three or more CDS features overlap;
  - ignores strand, mixing same-strand with opposite-strand overlap.
And `n_genes_eff = n_gene or n_cds` makes the fallback frequency unrecoverable from the
published TSV (n_gene=0 with n_cds=20 is indistinguishable from a real n_gene=20).

This script re-extracts from the GenBank flat files, recording per-record provenance of the
gene count and both overlap definitions, so that the frequency of the fallback and the
sensitivity of the target become measurable.

Usage:
    python reextract_annotation.py --gbff-dir <dir> --out ../../results/tables
"""
import os, sys, gzip, json, argparse, time
import numpy as np
import pandas as pd


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def union_len(spans):
    if not spans:
        return 0
    spans = sorted(spans)
    tot = 0; cs, ce = spans[0]
    for s, e in spans[1:]:
        if s > ce:
            tot += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def per_record(rec):
    L = len(rec.seq)
    cds_plus, cds_minus, n_gene, n_cds = [], [], 0, 0
    for f in rec.features:
        if f.type == "gene":
            n_gene += 1
        elif f.type == "CDS":
            n_cds += 1
            s, e = int(f.location.start), int(f.location.end)
            (cds_plus if f.location.strand != -1 else cds_minus).append((s, e))
    allc = cds_plus + cds_minus

    # as published: sum of spans minus union, strand-blind, double counts 3+ overlaps
    pub = max(0, sum(e - s for s, e in allc) - union_len(allc))

    # correct: positions with coverage >= 2, and the split by strand
    def cov(spans):
        c = np.zeros(L + 1, np.int16)
        for s, e in spans:
            c[max(s, 0):min(e, L)] += 1
        return c[:L]
    cp, cm = cov(cds_plus), cov(cds_minus)
    tot = cp + cm
    return {
        "accession": rec.id,
        "n_gene_features": n_gene, "n_CDS": n_cds,
        "used_gene_fallback": bool(n_gene == 0 and n_cds > 0),
        "overlap_bp_published": int(pub),
        "overlap_positions": int((tot >= 2).sum()),
        "overlap_same_strand": int(((cp >= 2) | (cm >= 2)).sum()),
        "overlap_opposite_strand": int(((cp >= 1) & (cm >= 1)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", required=True)
    ap.add_argument("--out", default="results/json")
    ap.add_argument("--accessions", default=None,
                    help="optional TSV with an 'accession' column to restrict the report.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from Bio import SeqIO
    rows, n = [], 0
    op = gzip.open if a.gbff.endswith(".gz") else open
    with op(a.gbff, "rt") as fh:
        for rec in SeqIO.parse(fh, "genbank"):
            try:
                rows.append(per_record(rec))
            except Exception as e:
                log(f"  falhou {getattr(rec,'id','?')}: {type(e).__name__}")
            n += 1
            if n % 2000 == 0:
                log(f"  {n} records")
            if a.limit and n >= a.limit:
                break
    df = pd.DataFrame(rows)
    log(f"records processados: {len(df)}")

    scope = {"all_parsed": df}
    if a.accessions:
        want = set(pd.read_csv(a.accessions, sep="\t")["accession"].astype(str))
        sub = df[df.accession.isin(want)]
        log(f"CONTROLE POSITIVO — pedidos {len(want)}, achados no gbff {len(sub)}")
        scope["analysed_subset"] = sub

    out = {"n_records_parsed": int(len(df)), "scopes": {}}
    for name, d in scope.items():
        if not len(d):
            continue
        chg = d["overlap_bp_published"] != d["overlap_positions"]
        out["scopes"][name] = {
            "n": int(len(d)),
            "gene_fallback": {
                "n_used": int(d["used_gene_fallback"].sum()),
                "pct": float(100 * d["used_gene_fallback"].mean()),
                "note": "records sem NENHUMA feature `gene`, em que n_genes caiu para n_CDS"},
            "overlap_definition": {
                "n_differs_from_published": int(chg.sum()),
                "pct_differs": float(100 * chg.mean()),
                "mean_published": float(d["overlap_bp_published"].mean()),
                "mean_positions": float(d["overlap_positions"].mean()),
                "mean_ratio_published_over_positions": float(
                    (d["overlap_bp_published"] / d["overlap_positions"].replace(0, np.nan)).mean()),
                "mean_same_strand": float(d["overlap_same_strand"].mean()),
                "mean_opposite_strand": float(d["overlap_opposite_strand"].mean()),
                "pct_of_overlap_that_is_opposite_strand": float(
                    100 * d["overlap_opposite_strand"].sum() /
                    max(d["overlap_positions"].sum(), 1))},
        }

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "annotation_provenance.json")
    json.dump(out, open(dst, "w"), indent=1)
    tsv_dir = os.path.abspath(os.path.join(a.out, "..", "tables"))
    os.makedirs(tsv_dir, exist_ok=True)
    df.to_csv(os.path.join(tsv_dir, "annotation_provenance.tsv.gz"),
              sep="\t", index=False, compression="gzip")

    for name, s in out["scopes"].items():
        o = s["overlap_definition"]
        log(f"\n=== {name} (n={s['n']}) ===")
        log(f"  fallback gene->CDS: {s['gene_fallback']['n_used']} "
            f"({s['gene_fallback']['pct']:.1f}%)")
        log(f"  overlap difere da definicao publicada em {o['n_differs_from_published']} "
            f"records ({o['pct_differs']:.1f}%)")
        log(f"  media publicada={o['mean_published']:.1f} vs posicoes={o['mean_positions']:.1f} "
            f"(razao {o['mean_ratio_published_over_positions']:.3f})")
        log(f"  do overlap real, {o['pct_of_overlap_that_is_opposite_strand']:.1f}% "
            f"e entre fitas OPOSTAS")
    log(f"\n-> {dst}")


if __name__ == "__main__":
    main()
