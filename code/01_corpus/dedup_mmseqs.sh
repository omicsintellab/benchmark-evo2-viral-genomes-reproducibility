#!/usr/bin/env bash
# dedup_mmseqs.sh — Fase 4: deduplicação intra-corpus com MMseqs2 linclust.
# Colapsa sequências >= min_seq_id (cov cov_mode) num representante -> remove redundância
# (cepas quase-idênticas do RefSeq) ANTES do split, evitando vazamento train/test.
# Params do corpus_design.yaml: min_seq_id 0.95, coverage 0.85, cov_mode 1.
#
# Uso: ./dedup_mmseqs.sh <in.fasta> <out_prefix> [min_seq_id] [cov] [cov_mode]
set -euo pipefail
IN="${1:?in.fasta}"; OUT="${2:?out_prefix}"
ID="${3:-0.95}"; COV="${4:-0.85}"; COVMODE="${5:-1}"
TMP="$(mktemp -d)"
echo "[dedup] MMseqs2 linclust id=$ID cov=$COV cov_mode=$COVMODE sobre $IN"
mmseqs createdb "$IN" "${TMP}/db" >/dev/null
mmseqs linclust "${TMP}/db" "${TMP}/clu" "${TMP}/tmp" \
  --min-seq-id "$ID" -c "$COV" --cov-mode "$COVMODE" >/dev/null
mmseqs createsubdb "${TMP}/clu" "${TMP}/db" "${TMP}/repr" >/dev/null
mmseqs convert2fasta "${TMP}/repr" "${OUT}.fasta" >/dev/null
mmseqs createtsv "${TMP}/db" "${TMP}/db" "${TMP}/clu" "${OUT}_clusters.tsv" >/dev/null
N_IN=$(grep -c '^>' "$IN"); N_OUT=$(grep -c '^>' "${OUT}.fasta")
echo "[dedup] $N_IN -> $N_OUT representantes (${OUT}.fasta) | clusters: ${OUT}_clusters.tsv"
rm -rf "$TMP"
