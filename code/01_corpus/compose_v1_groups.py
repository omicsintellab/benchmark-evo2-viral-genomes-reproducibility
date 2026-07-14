#!/usr/bin/env python3
"""Compõe o V1 RefSeq: atribui a cada genoma um GRUPO DE COTA (Baltimore × host),
via join com a ICTV VMR. NÃO impõe as cotas (isso é a Fase 5/balanceamento) — apenas
etiqueta e separa, para o balanceamento amostrar dentro de cada grupo depois.

Entrada: o `genome_features.parquet` de extract_genome_features.py (que já traz
accession, family, genus, genome_length extraídos do gbff) — NÃO o JSONL do
`datasets summary` (fluxo antigo, aposentado quando o artefato primário virou o gbff).

Saídas em --out-dir:
  fasta/<grupo>.fasta            genomas roteados por grupo (+ unassigned.fasta)
  composition_manifest.parquet   1 linha/genoma: accession, family, baltimore, host, quota_group, is_recent, length
  provenance.tsv                 source, accession, version, download_date
  coverage_report.tsv            grupo -> n_disponível vs N_alvo (sinaliza sub-representação JÁ aqui)
"""
import argparse, csv, datetime as dt
from collections import defaultdict
import pandas as pd

# Baltimore (string da VMR "Genome composition") -> grupo de cota, dado o domínio do host.
BALT = {  # normalizado (lower, sem espaços/sinais) -> classe de Baltimore
    "dsdna": "I",
    "ssdna": "II", "ssdna(+)": "II", "ssdna(-)": "II", "ssdna(+/-)": "II",
    "dsrna": "III",
    "ssrna(+)": "IV", "ssrnapositive": "IV", "ssrna+": "IV",
    "ssrna(-)": "V", "ssrnanegative": "V", "ssrna-": "V",
    "ssrna-rt": "VI", "dsdna-rt": "VII",
}

def norm(s):
    return "".join(str(s).lower().split()).replace("‐", "-")

def quota_group(balt_label, host_domain):
    """Regra de atribuição (ver NCBI_REFSEQ_ACQUISITION.md §1)."""
    if host_domain == "bacteria":   return "phage"
    if host_domain == "archaea":    return "archaeal_virus"
    if balt_label in ("VI", "VII"): return "retro"
    if host_domain == "eukaryote":
        return {"I": "dsDNA_euk", "II": "ssDNA_euk", "III": "dsRNA_euk",
                "IV": "ssRNA_pos_euk", "V": "ssRNA_neg_euk"}.get(balt_label,
               f"unassigned_{balt_label}_euk")   # ex.: ssRNA ambisense (?) sem cota -> auditar
    return "unassigned"

def host_domain(vmr_host):
    h = norm(vmr_host)
    if "bacteria" in h:           return "bacteria"
    if "archaea" in h:            return "archaea"
    if not h or h in ("nan",):    return "unknown"
    return "eukaryote"            # plants/vertebrates/invertebrates/fungi/protists/algae/human

def _read_vmr_sheet(path):
    """Lê a aba de DADOS da VMR — a que tem colunas Family/Genus, não a capa 'Version'.
    A VMR MSLxx tem múltiplas abas (Version, VMR MSLxx, Column definitions, ...); a
    primeira (lida por pd.read_excel default) é a capa e não tem os dados."""
    xls = pd.ExcelFile(path)
    for sh in xls.sheet_names:
        head = {str(c).lower().strip() for c in pd.read_excel(xls, sheet_name=sh, nrows=0).columns}
        if "family" in head and "genus" in head:
            return pd.read_excel(xls, sheet_name=sh)
    raise SystemExit(f"VMR sem aba de dados (Family/Genus). Abas: {xls.sheet_names}")

def family_set(path):
    """Conjunto de nomes de família presentes numa VMR/MSL (para o diff de ictv_recent)."""
    df = _read_vmr_sheet(path)
    fam_c = {str(c).lower().strip(): c for c in df.columns}.get("family")
    return set(str(f).strip() for f in df[fam_c].dropna()) if fam_c else set()

def load_vmr(path, recent_fams=frozenset()):
    """família/gênero -> (balt_label, host_domain, is_recent).
    is_recent vem do diff de famílias (família ∈ recent_fams), não de coluna na VMR —
    a VMR é um snapshot e não carrega o MSL de criação da família."""
    df = _read_vmr_sheet(path)
    cols = {str(c).lower().strip(): c for c in df.columns}
    fam_c  = cols.get("family"); gen_c = cols.get("genus")
    # Baltimore: na VMR a coluna se chama 'Genome' (valores dsDNA, ssRNA(+), ...).
    balt_c = cols.get("genome") or cols.get("genome composition") or cols.get("genome_composition")
    host_c = cols.get("host source") or cols.get("host")
    by_family, by_genus = {}, {}
    for _, r in df.iterrows():
        balt = BALT.get(norm(r.get(balt_c, "")), "?")
        hd   = host_domain(r.get(host_c, ""))
        fam  = str(r[fam_c]).strip() if fam_c and pd.notna(r.get(fam_c)) else ""
        # ictv_recent restrito a vírus de EUCARIOTOS: o objetivo é remendar o gap
        # eucariótico do EVO2 base; famílias novas de fagos (expansão Caudoviricetes)
        # não servem a esse fim e dominariam o grupo.
        recent = (fam in recent_fams) and (hd == "eukaryote")
        if fam:                              by_family.setdefault(fam, (balt, hd, recent))
        if gen_c and pd.notna(r.get(gen_c)): by_genus.setdefault(str(r[gen_c]).strip(), (balt, hd, recent))
    return by_family, by_genus

def iter_fasta(path):
    acc, buf = None, []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(">"):
                if acc is not None: yield acc, "".join(buf)
                acc = ln[1:].split()[0]; buf = []
            else: buf.append(ln)
        if acc is not None: yield acc, "".join(buf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="genome_features.parquet de extract_genome_features.py")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--vmr", required=True, help="VMR atual (snapshot da taxonomia corrente)")
    ap.add_argument("--vmr-baseline", required=False,
                    help="VMR/MSL antiga (ex. MSL38) p/ derivar ictv_recent por diff de famílias")
    ap.add_argument("--design", required=False, help="corpus_design.yaml (fonte única de cotas/cortes)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import os; os.makedirs(os.path.join(a.out_dir, "fasta"), exist_ok=True)

    # cotas vêm do corpus_design.yaml (fonte única). Fallback aos defaults só se --design
    # não for passado — mas em produção, SEMPRE passar.
    targets = dict(dsDNA_euk=800, ssDNA_euk=500, ssRNA_pos_euk=1500, ssRNA_neg_euk=1000, dsRNA_euk=300,
                   retro=400, phage=2000, archaeal_virus=200, ictv_recent=800)
    if a.design:
        import yaml
        v1 = (yaml.safe_load(open(a.design)) or {}).get("v1_refseq", {})
        targets = v1.get("quotas", targets)
    else:
        print("AVISO: --design não passado; usando cotas default embutidas (não pré-registradas).")

    # ictv_recent = famílias na VMR atual e AUSENTES na baseline (criadas em MSLs recentes).
    recent_fams = frozenset()
    if a.vmr_baseline:
        recent_fams = frozenset(family_set(a.vmr) - family_set(a.vmr_baseline))
        print(f"ictv_recent: {len(recent_fams)} famílias novas (em {os.path.basename(a.vmr)} e não na baseline)")
    else:
        print("AVISO: --vmr-baseline não passado; ictv_recent ficará 0.")

    by_family, by_genus = load_vmr(a.vmr, recent_fams=recent_fams)
    today = dt.date.today().isoformat()

    # 1) atribuir grupo por genoma a partir da tabela de features (extraída do gbff)
    feat = pd.read_parquet(a.features)
    rec = {}   # accession (sem versão) -> dict
    for row in feat.itertuples(index=False):
        acc = str(getattr(row, "accession", "") or "")
        if not acc: continue
        fam = str(getattr(row, "family", "") or "")
        gen = str(getattr(row, "genus", "") or "")
        length = int(getattr(row, "genome_length", 0) or 0)
        balt, hd, recent = by_family.get(fam) or by_genus.get(gen) or ("?", "unknown", False)
        grp = quota_group(balt, hd)
        rec[acc.split(".")[0]] = dict(accession=acc, family=fam, genus=gen, baltimore=balt,
                                      host=hd, quota_group=grp, is_recent=recent, length=length)

    # 2) rotear FASTA por grupo (chave = accession sem versão)
    writers, counts = {}, defaultdict(int)
    prov = open(os.path.join(a.out_dir, "provenance.tsv"), "w", newline="")
    pw = csv.writer(prov, delimiter="\t"); pw.writerow(["source","accession","version","download_date"])
    for header_acc, seq in iter_fasta(a.fasta):
        base = header_acc.split(".")[0]
        meta = rec.get(base, dict(quota_group="unassigned", accession=header_acc))
        grp = meta["quota_group"]
        fpath = os.path.join(a.out_dir, "fasta", f"{grp}.fasta")
        w = writers.get(grp) or writers.setdefault(grp, open(fpath, "w"))
        w.write(f">{header_acc}\n{seq if seq.endswith(chr(10)) else seq+chr(10)}")
        counts[grp] += 1
        pw.writerow(["NCBI_RefSeq", header_acc, header_acc.split('.')[-1] if '.' in header_acc else "", today])
    for w in writers.values(): w.close()
    prov.close()

    # 3) manifesto + relatório de cobertura (cotas de `targets`, vindo do corpus_design.yaml)
    pd.DataFrame(rec.values()).to_parquet(os.path.join(a.out_dir, "composition_manifest.parquet"), index=False)
    with open(os.path.join(a.out_dir, "coverage_report.tsv"), "w", newline="") as fh:
        cw = csv.writer(fh, delimiter="\t"); cw.writerow(["quota_group","n_disponivel","n_alvo","status"])
        for g, tgt in targets.items():
            n = counts.get(g, 0) if g != "ictv_recent" else sum(1 for r in rec.values() if r["is_recent"])
            cw.writerow([g, n, tgt, "OK" if n >= tgt else "SUB-REPRESENTADO"])
        for g, n in sorted(counts.items()):
            if g not in targets: cw.writerow([g, n, "-", "extra/auditar"])
    print("composição pronta:", dict(counts))
    print("→ revise coverage_report.tsv (grupos SUB-REPRESENTADOS e unassigned*) antes de seguir p/ §3.")

if __name__ == "__main__":
    main()
