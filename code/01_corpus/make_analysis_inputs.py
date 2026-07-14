#!/usr/bin/env python3
"""make_analysis_inputs.py — prepara os inputs de análise a partir do que é versionado.

O repositório versiona o corpus em TSV comprimido (legível, inspecionável em qualquer
ferramenta). Os scripts das etapas 02/03/04 consomem parquet, com os nomes usados na
execução original. Este script faz a ponte, sem duplicar o dado no git:

    data/corpus_manifest.tsv.gz   ->  <workdir>/manifest.parquet
    data/genome_features.tsv.gz   ->  <workdir>/features.parquet
    data/cl95_cluster.tsv         ->  <workdir>/cl95_cluster.tsv  (cópia)

Uso:
    python code/01_corpus/make_analysis_inputs.py [--workdir DIR]

O FASTA (all_genomes.fasta) NÃO é gerado aqui — ele não é versionado (tamanho); veja
data/README.md para reconstruí-lo a partir da lista de accessions.
"""
import argparse, os, shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
DATA = os.path.join(REPO, "data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default=DATA,
                   help="Destino dos parquets (default: data/, que é o SCALE_DATA default).")
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    for src, dst in [("corpus_manifest.tsv.gz", "manifest.parquet"),
                     ("genome_features.tsv.gz", "features.parquet")]:
        df = pd.read_csv(os.path.join(DATA, src), sep="\t")
        out = os.path.join(a.workdir, dst)
        df.to_parquet(out, index=False)
        print(f"[ok] {dst}: {len(df):,} linhas -> {out}")

    clu = os.path.join(DATA, "cl95_cluster.tsv")
    if os.path.abspath(a.workdir) != os.path.abspath(DATA):
        shutil.copy(clu, a.workdir)
        print(f"[ok] cl95_cluster.tsv -> {a.workdir}")


if __name__ == "__main__":
    main()
