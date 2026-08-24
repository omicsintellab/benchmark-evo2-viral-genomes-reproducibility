#!/usr/bin/env python3
"""enc_summary.py — resumo do numero efetivo de codons (ENC) para o corpo do texto (R2 #18).

R2 #18: o ENC e citado na Introducao e nunca reportado. Ele SEMPRE existiu -- foi computado
com o subconjunto de features -- mas vivia so no `codon_enc_cds.parquet` do S3, que nao e
versionado. Este script destila o parquet num JSON pequeno (mediana, IQR, faixa, n) global e
por classe de Baltimore, que e o que entra no manuscrito e o que o repositorio publico pode
carregar sem distribuir dado bruto.

O ENC vai de 20 (uso maximamente enviesado: um codon por aminoacido) a 61 (uso uniforme).

Uso:
    python code/03_analysis/enc_summary.py \
        --enc <dir>/codon_enc_cds.parquet --manifest <dir>/composition_manifest.parquet \
        --out results/json/enc_summary.json
"""
import argparse, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def block(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return {"n": int(s.size), "median": round(float(s.median()), 1),
            "iqr": [round(float(q1), 1), round(float(q3), 1)],
            "range": [round(float(s.min()), 1), round(float(s.max()), 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "json", "enc_summary.json"))
    a = ap.parse_args()

    enc = pd.read_parquet(a.enc)
    man = pd.read_parquet(a.manifest).set_index("accession")
    cov = man.reindex(enc.index)["baltimore"]
    if cov.isna().any():
        raise SystemExit(f"{int(cov.isna().sum())} records sem classe de Baltimore no manifesto")

    v = enc["codon_enc_cds"].astype(float)
    out = {"source": "codon_enc_cds.parquet (S3), destilado — o parquet nao e versionado",
           "n_records": int(len(enc)), "feature_subset_n": 1200,
           "coverage_note": f"{len(enc)}/1200 records do subconjunto de features",
           "scale_note": "ENC vai de 20 (uso de codons maximamente enviesado) a 61 (uniforme)",
           "overall": block(v),
           "by_baltimore": {str(k): block(g) for k, g in v.groupby(cov.values)},
           "median_codons_per_record": int(enc["n_codons_cds"].median())}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"-> {a.out}")
    print(f"   global: mediana {out['overall']['median']} "
          f"IQR {out['overall']['iqr']} faixa {out['overall']['range']}")
    for k, b in out["by_baltimore"].items():
        print(f"   {k:14s} n={b['n']:4d} mediana={b['median']} IQR={b['iqr']}")


if __name__ == "__main__":
    main()
