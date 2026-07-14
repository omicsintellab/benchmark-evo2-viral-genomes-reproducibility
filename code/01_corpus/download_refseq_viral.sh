#!/usr/bin/env bash
# Aquisição do V1 — RefSeq viral em GenBank (.gbff) + ICTV VMR.
# GenBank = artefato primário (sequência + FEATURES + taxonomia num só arquivo).
# Pré: wget, gzip. Saída em $OUT (default /data/v1_raw/).
set -euo pipefail

OUT=${OUT:-/data/v1_raw}
FTP=${FTP:-https://ftp.ncbi.nlm.nih.gov/refseq/release/viral}
VMR_URL=${VMR_URL:-https://ictv.global/sites/default/files/VMR/VMR_MSL41.v1.20260320.xlsx}  # MSL41 (2026-03-20); conferir MSL atual em ictv.global/vmr
mkdir -p "$OUT/genbank" "$OUT/features" "$OUT/fasta"
cd "$OUT/genbank"

echo "[1/3] Listando e baixando os GenBank flat files do RefSeq release viral"
# O release viral é dividido em poucos arquivos viral.N.genomic.gbff.gz. Baixa todos.
wget -q -O - "$FTP/" \
  | grep -oE 'viral\.[0-9]+\.genomic\.gbff\.gz' | sort -u > files.txt
echo "    arquivos a baixar: $(wc -l < files.txt)"
while read -r f; do
  echo "    -> $f"
  wget -q -c "$FTP/$f"
done < files.txt
echo "    gbff baixados: $(ls viral.*.genomic.gbff.gz 2>/dev/null | wc -l)"

echo "[2/3] ICTV VMR (família → Baltimore + host + MSL)"
wget -q -O "$OUT/ictv_vmr.xlsx" "$VMR_URL" || \
  echo "    !! falha no VMR_URL; baixe a VMR atual de https://ictv.global/vmr e salve em $OUT/ictv_vmr.xlsx"

echo "[3/3] Snapshot/proveniência"
{
  echo "snapshot_date: $(date -u +%FT%TZ)"
  echo "refseq_release_ftp: $FTP"
  echo "n_gbff_files: $(ls viral.*.genomic.gbff.gz 2>/dev/null | wc -l)"
  echo "vmr_url: $VMR_URL"
} > "$OUT/README_acquisition.txt"
cat "$OUT/README_acquisition.txt"

echo "OK — gbff em $OUT/genbank. Próximo: python code/extract_genome_features.py"
