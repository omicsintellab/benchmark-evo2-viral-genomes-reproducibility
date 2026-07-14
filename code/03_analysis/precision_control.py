#!/usr/bin/env python3
"""precision_control.py — Controle de precisão FP8 vs bf16 para a comparação de escala.

O sweep_layers_20b.py (CV simples, R2_inc) mostrou uma queda GRANDE (~14pp em
Baltimore) ao forçar bf16 em blocks.15/18 do 20B, maior do que se espera de um
mero controle de precisão. Antes de aceitar isso como efeito real, reproduz a
comparação com a MESMA metodologia rigorosa do scale_analysis.py (R2(emb) cru,
baselines 6-mer/GC+len, CV cluster-aware, 3 repeats) — só assim os números são
comparáveis à Tabela final de escala.

Reusa os embeddings já extraídos (fp8 e bf16) e os helpers de scale_analysis.py.
Roda em CPU; independente do scale_analysis.py.

Uso: python precision_control.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_analysis import (load_meta, load_7b, load_20b, align, groups,
        clf_rand, reg_rand, clf_group, reg_group, REG_FEATS, OUTDIR)

LAYERS = [15, 18]

def suite_clf(X, y, g):
    return {"rand": clf_rand(X, y), "clus": clf_group(X, y, g)}
def suite_reg(X, y, g):
    return {"rand": reg_rand(X, y), "clus": reg_group(X, y, g)}

def main():
    X7b_b, accb = load_7b("baltimore")
    X7b_f, accf = load_7b("features")
    db, dff = load_meta(accb), load_meta(accf)
    gb, gf = groups(accb), groups(accf)
    hm = dff["host"].isin(["eukaryote", "bacteria", "archaea"]).values
    fam = dff["family"].replace("", np.nan); topf = fam.value_counts(); topf = topf[topf >= 25].index.tolist()
    fk = dff["family"].isin(topf).values

    def run_rep(name, Xb, Xf):
        print(f"[precision_control] {name} ...", flush=True)
        r = {}
        r["Baltimore"] = suite_clf(Xb, db["baltimore"].values, gb)
        r["Host"] = suite_clf(Xf[hm], dff["host"].values[hm], gf[hm])
        r["Family"] = suite_clf(Xf[fk], dff["family"].values[fk], gf[fk])
        for f, logt in REG_FEATS:
            y = dff[f].values.astype(float); y = np.log1p(np.clip(y, 0, None)) if logt else y
            r[f] = suite_reg(Xf, y, gf)
        print(f"  Baltimore (clus): {r['Baltimore']['clus'].mean():.3f}", flush=True)
        return r

    out = {"layers": LAYERS, "reps": {}}
    for L in LAYERS:
        for prec in ["fp8", "bf16"]:
            Xb20, ab20 = load_20b(L, prec); Xf20, af20 = load_20b(L, prec)
            if Xb20 is None:
                print(f"[aviso] L{L} {prec} ausente, pulando"); continue
            Xb_al, kb = align(Xb20, ab20, accb); Xf_al, kf = align(Xf20, af20, accf)
            if not (kb.all() and kf.all()):
                print(f"[aviso] L{L} {prec}: cobertura incompleta, pulando"); continue
            out["reps"][f"blocks{L}_{prec}"] = run_rep(f"blocks{L}_{prec}", Xb_al, Xf_al)

    def dump(d):
        if isinstance(d, dict): return {k: dump(v) for k, v in d.items()}
        if isinstance(d, np.ndarray): return {"mean": float(d.mean()), "std": float(d.std())}
        return d
    json.dump(dump(out), open(f"{OUTDIR}/precision_control_metrics.json", "w"), indent=2)
    print("[ok] precision_control_metrics.json", flush=True)

    print(f"\n{'target':<20}", end="")
    for L in LAYERS:
        print(f"{'bl'+str(L)+'_fp8':>14}{'bl'+str(L)+'_bf16':>15}", end="")
    print()
    targets = ["Baltimore", "Host", "Family"] + [f for f, _ in REG_FEATS]
    for t in targets:
        print(f"{t:<20}", end="")
        for L in LAYERS:
            for prec in ["fp8", "bf16"]:
                key = f"blocks{L}_{prec}"
                v = out["reps"].get(key, {}).get(t, {}).get("clus", {}).get("mean")
                print(f"{v if v is None else round(v,3):>14}" if prec == "fp8" else f"{v if v is None else round(v,3):>15}", end="")
        print()

if __name__ == "__main__":
    main()
