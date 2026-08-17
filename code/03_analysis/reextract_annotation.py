#!/usr/bin/env python3
"""reextract_annotation.py — proveniência da anotação e sobreposição corrigida (R1 4.1, R2 #16/#17).

R2 #16: "substituir contagem de genes por contagem de CDS quando features `gene` estão
ausentes cria uma definição mista. Os autores devem reportar com que frequência esse
fallback foi usado e fornecer uma análise de sensibilidade restrita a genomas
consistentemente anotados."

R2 #17: "'comprimento somado de CDS menos comprimento da união' nem sempre é idêntico ao
número de posições cobertas por mais de um CDS. O cálculo implementado e o tratamento de
fitas opostas e de sobreposições múltiplas devem ser descritos com precisão."

Ambos são procedentes. O `extract_genome_features.py` publicado faz
`overlap_bp = soma(spans) - união`, sobre spans coletados no ramo `t == "CDS"`, o que:
  - conta em DOBRO onde 3+ CDS se sobrepõem;
  - ignora fita, misturando sobreposição na mesma fita com fitas opostas.
E `n_genes_eff = n_gene or n_cds` torna a frequência do fallback irrecuperável do TSV
publicado (n_gene=0 com n_cds=20 é indistinguível de n_gene=20 real).

Este script re-extrai do GenBank flat file registrando:
  - `used_gene_fallback`  — se o record não tinha nenhuma feature `gene`
  - `overlap_bp_published` — a métrica como publicada, para comparação
  - `overlap_positions`    — posições cobertas por >= 2 CDS (a definição correta)
  - `overlap_same_strand` / `overlap_opposite_strand`

Uso:
    python reextract_annotation.py --gbff viral.1.genomic.gbff.gz --out ../../results/json
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

    # como publicado: soma dos spans - união, sem fita, conta 3+ sobreposições em dobro
    pub = max(0, sum(e - s for s, e in allc) - union_len(allc))

    # correto: posições com cobertura >= 2, e a separação por fita
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
                    help="TSV opcional com coluna 'accession' para restringir o relatório.")
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
