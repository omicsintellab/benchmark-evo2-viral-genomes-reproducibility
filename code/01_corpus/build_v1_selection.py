#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v1_selection.py — Fases 2/3/5 do corpus V1 (RefSeq viral).

Aplica a POLÍTICA DE BALANCEAMENTO v2 (corpus_design.yaml -> v1_refseq.balancing):
  - keep_all: grupos euk + retro + archaeal_virus + unassigned_?_euk (todo o disponível);
  - cap: phage -> 2000 (priority sampling, seed do design);
  - exclude: unassigned (host unknown).
Sanitização (Fase 2/3): min_length, corte por fração de N, uppercase, alfabeto ACGTN.

Saídas: curated_v1.fasta + curated_v1_manifest.parquet (1 linha/sequência selecionada).
--dry-run: só conta a seleção a partir do manifesto (não lê o FASTA), p/ validar a composição.

Uso:
  python build_v1_selection.py --design ../corpus_design.yaml \
    --manifest composition_manifest.parquet --fasta all_genomes.fasta \
    --out-fasta curated_v1.fasta --out-manifest curated_v1_manifest.parquet
  python build_v1_selection.py --design ... --manifest ... --dry-run    # só números
"""
import argparse, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fasta", default=None)
    p.add_argument("--out-fasta", default="curated_v1.fasta")
    p.add_argument("--out-manifest", default="curated_v1_manifest.parquet")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    a = parse_args()
    import yaml, pandas as pd, numpy as np
    d = yaml.safe_load(open(a.design))
    v1 = d["v1_refseq"]; bal = v1["balancing"]
    min_len = int(v1.get("min_length", 1000))
    max_N = float(d["sanitize"]["max_N_fraction"])
    seed = int(d["split"]["seed"])
    keep_all = set(bal["keep_all"]); cap = dict(bal.get("cap", {})); exclude = set(bal.get("exclude", []))

    man = pd.read_parquet(a.manifest)
    print(f"manifesto: {len(man)} records | grupos: {sorted(man['quota_group'].unique())}")

    # 1) filtro de comprimento (a fração de N só no run completo, precisa da sequência)
    n0 = len(man)
    man = man[man["length"] >= min_len].copy()
    print(f"após min_length>={min_len}: {len(man)} (-{n0-len(man)})")

    # 2) aplicar a política de balanceamento por grupo
    rng = np.random.default_rng(seed)
    parts = []
    for grp, g in man.groupby("quota_group"):
        if grp in exclude:
            print(f"  [exclui] {grp}: {len(g)}"); continue
        if grp in cap:
            k = min(cap[grp], len(g))
            g = g.sample(k, random_state=seed)
            print(f"  [cap {cap[grp]}] {grp}: {len(g)}")
        elif grp in keep_all:
            print(f"  [keep_all] {grp}: {len(g)}")
        else:
            print(f"  [keep (default)] {grp}: {len(g)}")   # grupo não citado: manter (logado)
        parts.append(g)
    sel = pd.concat(parts).reset_index(drop=True)
    print(f"\nSELEÇÃO (pré-sanitização de N): {len(sel)} genomas")
    print("por host:"); print(sel["host"].value_counts().to_string())
    print("por baltimore:"); print(sel["baltimore"].value_counts().to_string())
    euk = (sel["host"] == "eukaryote").sum()
    print(f"eucariotos: {euk}/{len(sel)} = {100*euk/len(sel):.1f}% (alvo >= {v1.get('target_eukaryotic_pct')}%)")

    if a.dry_run:
        print("\n[dry-run] não escreve FASTA/manifesto. Composição acima é a esperada (sem corte de N).")
        return

    # 3) extrair as sequências do FASTA + sanitizar (uppercase, ACGTN, corte de N)
    assert a.fasta, "--fasta é obrigatório fora do --dry-run"
    wanted = set(sel["accession"])
    kept, dropped_N, dropped_len = {}, 0, 0
    acc = None; buf = []
    def flush():
        nonlocal dropped_N, dropped_len
        if acc is None or acc not in wanted: return
        s = "".join(buf).upper()
        s = "".join(c if c in "ACGTN" else "N" for c in s)   # alfabeto: não-ACGTN vira N
        if len(s) < min_len: dropped_len += 1; return
        if s.count("N")/len(s) > max_N: dropped_N += 1; return
        kept[acc] = s
    with open(a.fasta) as fh:
        for ln in fh:
            if ln.startswith(">"): flush(); acc = ln[1:].split()[0]; buf = []
            else: buf.append(ln.strip())
        flush()
    print(f"\nsequências extraídas e sanitizadas: {len(kept)} "
          f"(descartadas: {dropped_N} por N>{max_N}, {dropped_len} por <{min_len}bp)")

    # 4) escrever FASTA curado + manifesto alinhado
    with open(a.out_fasta, "w") as out:
        for acc in sel["accession"]:
            if acc in kept:
                out.write(f">{acc}\n")
                seq = kept[acc]
                for i in range(0, len(seq), 80): out.write(seq[i:i+80] + "\n")
    cur = sel[sel["accession"].isin(kept)].copy()
    cur["clean_length"] = cur["accession"].map(lambda x: len(kept[x]))
    cur.to_parquet(a.out_manifest, index=False)
    print(f"escrito: {a.out_fasta} ({len(cur)} seqs) + {a.out_manifest}")

if __name__ == "__main__":
    main()
