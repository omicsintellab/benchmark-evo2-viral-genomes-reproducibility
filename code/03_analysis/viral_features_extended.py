#!/usr/bin/env python3
"""viral_features_extended.py — Painel estendido de features virais pro 20B (blocks.18,
a camada principal do paper), com a MESMA suíte rigorosa (R2(emb) cru, baselines
6-mer/GC+len, CV cluster-aware, 3 repeats) usada em scale_analysis.py.

Duas features NOVAS, biologicamente motivadas (não estavam no painel original):

  - CpG / UpA dinucleotide depletion (observado/esperado): vírus de RNA de
    vertebrados sofrem supressão de CpG (reconhecido pelo sensor antiviral ZAP)
    e UpA; é marca de adaptação vírus-hospedeiro, não composição genérica.
    Calculada da sequência (genoma inteiro), ortogonal a GC.
  - Gene overlap (overlap_bp, já no genome_features.parquet): ORFs sobrepostas
    são compressão arquitetural característica de vírus pequenos (ssRNA/ssDNA).

Roda em CPU, reusa os embeddings já cacheados (7B e 20B blocks.18, fp8) e a
mesma amostra/CV de scale_analysis.py.

Uso: python viral_features_extended.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_analysis import (load_meta, load_7b, load_20b, align, groups, seqs_for,
        kmer_matrix, gclen, reg_rand, OUTDIR, DATA)
import pandas as pd

# ----------------------------------------------------------------------------- CpG/UpA o/e
def dinuc_oe(seq):
    """Observado/esperado p/ CpG e UpA (=TpA em DNA). O/E_XY = f(XY) / (f(X)*f(Y)),
    frequências relativas na própria sequência (genoma inteiro). Retorna (cpg_oe, upa_oe)."""
    s = "".join(c for c in seq.upper() if c in "ACGT")
    n = len(s)
    if n < 100: return np.nan, np.nan
    from collections import Counter
    mono = Counter(s)
    di = Counter(s[i:i+2] for i in range(n-1))
    ntot = n - 1
    fA, fC, fG, fT = (mono.get(b, 0)/n for b in "ACGT")
    def oe(xy, fx, fy):
        fxy = di.get(xy, 0) / ntot
        exp = fx * fy
        return fxy / exp if exp > 0 else np.nan
    cpg_oe = oe("CG", fC, fG)
    upa_oe = oe("TA", fT, fA)
    return cpg_oe, upa_oe

def main():
    X7b_f, accf = load_7b("features")
    dff = load_meta(accf)
    gf = groups(accf)

    print("[viral_features_extended] calculando CpG/UpA o/e da sequencia ...", flush=True)
    seqs = seqs_for(accf.tolist())
    cpg, upa = [], []
    for a in accf:
        c, u = dinuc_oe(seqs.get(a, ""))
        cpg.append(c); upa.append(u)
    dff["cpg_oe"] = cpg
    dff["upa_oe"] = upa
    print(f"  CpG_oe: mean={np.nanmean(cpg):.3f} std={np.nanstd(cpg):.3f} "
          f"| UpA_oe: mean={np.nanmean(upa):.3f} std={np.nanstd(upa):.3f}", flush=True)

    # overlap_bp não faz parte do subset fixo de colunas de load_meta() -> ler direto do parquet
    fe_full = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")
    dff["overlap_bp"] = fe_full.reindex(accf)["overlap_bp"].values
    print(f"  overlap_bp cobertura: {(dff['overlap_bp'].fillna(0)!=0).mean()*100:.1f}%", flush=True)

    KMf, GLf = kmer_matrix(accf), gclen(dff)

    # 20B blocks.18 (principal) + 7B (nota) + baselines
    X20_18, ab20 = load_20b(18, "fp8")
    X20_18_al, k20 = align(X20_18, ab20, accf)
    if not k20.all():
        print("[aviso] cobertura 20B_blocks18 incompleta"); sys.exit(1)

    targets = [("cpg_oe", False), ("upa_oe", False), ("overlap_bp", True)]
    reps = {"20B_blocks18": X20_18_al, "7B_blocks28": X7b_f, "6mer": KMf, "GC+len": GLf}

    out = {"targets": [t for t, _ in targets], "results": {}}
    for name, X in reps.items():
        print(f"[probe] {name} ...", flush=True)
        row = {}
        for f, logt in targets:
            y = dff[f].values.astype(float)
            m = np.isfinite(y)
            if m.sum() < 50:
                print(f"  {f}: n valido insuficiente ({m.sum()}), pulando"); continue
            yv = np.log1p(np.clip(y[m], 0, None)) if logt else y[m]
            Xm = X[m]
            sc = reg_rand(Xm, yv)
            row[f] = {"mean": float(sc.mean()), "std": float(sc.std()), "n": int(m.sum())}
            print(f"  {f}: R2={sc.mean():.3f}±{sc.std():.3f} (n={m.sum()})", flush=True)
        out["results"][name] = row

    json.dump(out, open(f"{OUTDIR}/viral_features_extended_metrics.json", "w"), indent=2)
    print("[ok] viral_features_extended_metrics.json", flush=True)

    print(f"\n{'target':<15}", end="")
    for name in reps: print(f"{name:>16}", end="")
    print()
    for f, _ in targets:
        print(f"{f:<15}", end="")
        for name in reps:
            v = out["results"].get(name, {}).get(f, {}).get("mean")
            print(f"{v if v is None else round(v,3):>16}", end="")
        print()

if __name__ == "__main__":
    main()
