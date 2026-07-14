#!/usr/bin/env python3
"""Extrai features genômicas dos GenBank flat files (.gbff) do RefSeq viral e
deriva o FASTA. As features são os metadados que serão correlacionados com os
embeddings do Evo2 (ver 06_evo2_inference/BASELINE_EMBEDDING_PROBES.md).

Uma linha por RECORD (genoma ou segmento). Vírus segmentados → 1 linha/segmento,
com `accession` próprio; a agregação por organismo (se desejada) é feita depois.

Saídas:
  --out-features  parquet com 1 linha/record
  --out-fasta     FASTA derivado (id = accession)

Pré: biopython, pandas, pyarrow.
"""
import argparse, glob, gzip, statistics as st
import pandas as pd
from Bio import SeqIO
from Bio.Seq import UndefinedSequenceError
from Bio.SeqUtils import gc_fraction

CDS_LIKE = {"CDS"}
RNA_GENES = {"tRNA", "rRNA", "ncRNA"}

def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)

def union_len(intervals):
    """soma do comprimento da união de intervalos [(start,end),...] (0-based, end exclusivo)."""
    if not intervals: return 0
    iv = sorted(intervals); tot = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    return tot + (ce - cs)

def intergenic_lengths(spans, genome_len):
    """gaps entre spans gênicos ordenados (não-codificante entre genes)."""
    if not spans: return []
    iv = sorted(spans); gaps = []
    prev_end = iv[0][1]
    for s, e in iv[1:]:
        if s > prev_end: gaps.append(s - prev_end)
        prev_end = max(prev_end, e)
    return gaps

def feats_from_record(rec):
    L = len(rec.seq)
    cds, genes = [], []
    n_cds = n_gene = n_trna = n_rrna = n_ncrna = 0
    has_introns = False; plus = 0; total_strand = 0
    for f in rec.features:
        t = f.type
        if t == "CDS":
            n_cds += 1
            parts = list(f.location.parts)
            if len(parts) > 1: has_introns = True          # join() => splicing
            for p in parts: cds.append((int(p.start), int(p.end)))
            genes.append((int(f.location.start), int(f.location.end)))
            if f.location.strand is not None:
                total_strand += 1; plus += (f.location.strand == 1)
        elif t == "gene":
            n_gene += 1
        elif t == "tRNA": n_trna += 1
        elif t == "rRNA": n_rrna += 1
        elif t == "ncRNA": n_ncrna += 1

    coding_bp = union_len(cds)
    noncoding_bp = max(0, L - coding_bp)
    gaps = intergenic_lengths(genes, L)
    # taxonomia/host do próprio gbff
    ann = rec.annotations
    taxonomy = ann.get("taxonomy", []) or []
    family = next((t for t in taxonomy if t.endswith("viridae")), "")
    genus  = next((t for t in taxonomy if t.endswith("virus")), "")
    host = ""
    for f in rec.features:
        if f.type == "source":
            host = (f.qualifiers.get("host") or f.qualifiers.get("lab_host") or [""])[0]
            break
    n_genes_eff = n_gene or n_cds
    return dict(
        accession=rec.id, organism=ann.get("organism", ""),
        genome_length=L, GC=round(gc_fraction(rec.seq) * 100, 3),
        n_CDS=n_cds, n_genes=n_genes_eff, n_tRNA=n_trna, n_rRNA=n_rrna, n_ncRNA=n_ncrna,
        gene_density=round(n_genes_eff / (L / 1000), 4) if L else 0,
        mean_CDS_len=round(st.fmean([e - s for s, e in genes]), 1) if genes else 0,
        coding_bp=coding_bp, noncoding_bp=noncoding_bp,
        coding_fraction=round(coding_bp / L, 4) if L else 0,
        n_intergenic=len(gaps),
        mean_intergenic_len=round(st.fmean(gaps), 1) if gaps else 0,
        median_intergenic_len=round(st.median(gaps), 1) if gaps else 0,
        max_intergenic_len=max(gaps) if gaps else 0,
        has_introns=has_introns,
        overlap_bp=max(0, sum(e - s for s, e in genes) - union_len(genes)),
        strand_balance=round(plus / total_strand, 3) if total_strand else None,
        family=family, genus=genus, host=host,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff-glob", required=True, help="glob dos .gbff(.gz)")
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--out-fasta", required=True)
    a = ap.parse_args()

    rows = []
    n_skip = 0
    with open(a.out_fasta, "w") as fa:
        for path in sorted(glob.glob(a.gbff_glob)):
            print("parsing", path)
            with _open(path) as fh:
                for rec in SeqIO.parse(fh, "genbank"):
                    try:
                        seq = str(rec.seq)   # dispara UndefinedSequenceError em records CON (sem ORIGIN inline)
                    except UndefinedSequenceError:
                        n_skip += 1; continue
                    rows.append(feats_from_record(rec))
                    fa.write(f">{rec.id} {rec.annotations.get('organism','')}\n{seq}\n")
    df = pd.DataFrame(rows)
    df.to_parquet(a.out_features, index=False)
    print(f"OK — {len(df)} records | {n_skip} pulados (seq indefinida/CON) | features → {a.out_features} | fasta → {a.out_fasta}")
    # resumo rápido (sanity)
    if len(df):
        print(df[["genome_length","n_CDS","coding_fraction","GC"]].describe().round(2).to_string())

if __name__ == "__main__":
    main()
