#!/usr/bin/env python3
"""composition_baselines.py — baselines composicionais fortes (revisão R1 3.4, R2 #3).

R1 3.4: "uma tabela de frequência de 6-mer é apenas UMA forma de descrever composição.
Outras features simples — k-mers mais longos, uso de códons, número e comprimento de ORFs,
viés de fita, repetições — podem explicar parte do mesmo sinal."
R2 #3: "k-mers globais não retêm estrutura de ORF, periodicidade de quadro de leitura,
distribuição de stop codons, espaçamento gênico nem ordem de longo alcance. Controles
derivados de sequência mais fortes — contagem de ORFs, maior ORF, cobertura por ORF,
densidade de stop codons e potencial codificante em 6 quadros — devem ser incluídos."

Este script constrói exatamente isso e re-roda os probes. A pergunta que ele responde:
**o Evo 2 continua à frente quando o baseline deixa de ser um 6-mer e passa a ser tudo o que
se consegue extrair da sequência sem modelo?**

Representações:
  gc_len      GC + log(comprimento)                          — referência do artigo
  kmer{3..6}  frequências de k-mer                           — "vários k" (R1 3.4)
  multik      concatenação normalizada de k=1..6             — "multi-k combinado" (R1 3.4)
  codon       64 frequências de códon, em quadro, dos ORFs   — "uso de códons" (R1 3.4)
  dicodon     4096 frequências de dicódon, em quadro         — 6-mer COM quadro de leitura
  orf         features de ORF e potencial codificante        — R2 #3
  compo_all   multik + codon + orf                           — o baseline mais forte possível
  evo2        Evo 2 20B blocks.18                            — o que está sendo testado

`dicodon` é o contraste mais afiado do conjunto: tem a mesma dimensionalidade do 6-mer e a
mesma ordem de composição, mas **em quadro de leitura**. Se o 6-mer perde e o dicodon ganha, o
que faltava ao baseline era quadro, não comprimento de k.

Esquemas de CV: `cl95` (pré-registrado, primário) e `family` (interpretativo). Mecânica de
fold importada de family_cv.py — os testes pareados só valem se os folds forem os mesmos.

Nota de cobertura: k>6 NÃO é rodado. k=7 são 16.384 dimensões e o custo do ridge dominaria a
corrida sem acrescentar ao argumento, já que `dicodon` cobre a mesma escala com quadro. Isso é
limitação declarada, não omissão.

Uso:
    PYTHONPATH=<repro>/code/03_analysis python composition_baselines.py --out ../../results/json
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scale_analysis import DATA, load_20b, kmer_matrix, gclen, seqs_for, groups as cl95_groups
from family_cv import (PRIMARY, NEGCTRL, reg_group, reg_rand, nadeau_bengio, boot_ci, holm,
                       SPLITS, REPEATS)
from viral_features_extended import dinuc_oe

# --- as DUAS CLASSES de baseline -------------------------------------------------------
# R1 3.4 pede baselines de *composição de sequência*. Um scan de ORF não é composição: é uma
# versão simplificada do próprio pipeline de anotação que produziu os alvos (n_orfs_per_kb
# prevendo n_genes é quase a mesma medida por outra via). Misturar os dois numa coluna só de
# "melhor baseline" responde a pergunta errada, então eles são reportados separados.
#
# codon/dicodon ficam na classe COMPOSICIONAL embora exijam um scan de 6 quadros para definir
# quadro de leitura. É a escolha CONSERVADORA: fortalece a classe composicional e torna mais
# difícil para o Evo 2 vencê-la.
CLASSES = {
    "compositional": ["kmer3", "kmer4", "kmer5", "kmer6", "multik", "codon", "dicodon", "gc_len"],
    "annotation_like": ["orf", "compo_all"],
}

B = {"A": 0, "C": 1, "G": 2, "T": 3}
STOPS = {48, 50, 56}           # TAA, TAG, TGA em código base-4 (A0 C1 G2 T3)
ATG = 14
MIN_ORF_BP = 300


def encode(seq):
    a = np.frombuffer(seq.upper().encode("ascii", "ignore"), dtype=np.uint8)
    code = np.full(a.shape, -1, np.int8)
    for b, v in B.items():
        code[a == ord(b)] = v
    return code


def revcomp_code(c):
    out = c[::-1].copy()
    m = out >= 0
    out[m] = 3 - out[m]
    return out


def codons_of(code, frame):
    c = code[frame:]
    n = (len(c) // 3) * 3
    if n < 3:
        return np.empty(0, np.int32), np.empty(0, bool)
    c = c[:n].reshape(-1, 3)
    ok = (c >= 0).all(axis=1)
    idx = np.where(ok, c[:, 0] * 16 + c[:, 1] * 4 + c[:, 2], -1).astype(np.int32)
    return idx, ok


def orf_features(seq):
    """ORFs em 6 quadros + densidade de stop + uso de códon/dicódon em quadro."""
    code = encode(seq)
    L = max(len(code), 1)
    rc = revcomp_code(code)
    n_orfs = 0; tot_orf_bp = 0; longest = 0; lens = []
    stop_dens = []
    codon_cnt = np.zeros(64, np.float64)
    dicodon_cnt = np.zeros(4096, np.float64)
    for strand_code in (code, rc):
        for frame in (0, 1, 2):
            idx, ok = codons_of(strand_code, frame)
            if idx.size == 0:
                stop_dens.append(0.0); continue
            is_stop = np.isin(idx, list(STOPS))
            stop_dens.append(float(is_stop.sum()) / max(idx.size, 1))
            # ORFs = trecho entre stops que contenha um ATG
            stop_pos = np.flatnonzero(is_stop)
            bounds = np.concatenate(([-1], stop_pos, [idx.size]))
            for s, e in zip(bounds[:-1], bounds[1:]):
                seg = idx[s + 1:e]
                if seg.size == 0:
                    continue
                atg = np.flatnonzero(seg == ATG)
                if atg.size == 0:
                    continue
                orf = seg[atg[0]:]
                bp = orf.size * 3
                if bp < MIN_ORF_BP:
                    continue
                n_orfs += 1; tot_orf_bp += bp; lens.append(bp)
                longest = max(longest, bp)
                good = orf[(orf >= 0)]
                codon_cnt += np.bincount(good, minlength=64)[:64]
                if good.size > 1:
                    di = good[:-1] * 64 + good[1:]
                    dicodon_cnt += np.bincount(di, minlength=4096)[:4096]
    cs = codon_cnt.sum(); ds = dicodon_cnt.sum()
    feats = {
        "n_orfs_per_kb": n_orfs / (L / 1000.0),
        "log_longest_orf": np.log1p(longest),
        "orf_coverage": min(tot_orf_bp / L, 6.0),
        "mean_orf_len": float(np.mean(lens)) if lens else 0.0,
        "sd_orf_len": float(np.std(lens)) if len(lens) > 1 else 0.0,
        "log_len": np.log1p(L),
    }
    for i, d in enumerate(stop_dens):
        feats[f"stop_dens_f{i}"] = d
    return (feats,
            codon_cnt / cs if cs else codon_cnt,
            dicodon_cnt / ds if ds else dicodon_cnt)


def build_reps(accs, md):
    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
    seqs = seqs_for(list(accs))
    log(f"CONTROLE POSITIVO — accessions: {len(accs)} | sequências achadas: {len(seqs)}")
    if len(seqs) < len(accs):
        raise SystemExit(f"FASTA incompleto: faltam {len(accs)-len(seqs)}")

    reps = {}
    for k in (3, 4, 5, 6):
        log(f"  k-mer k={k} ({4**k} dims) ...")
        import scale_analysis as SA
        old = SA.K; SA.K = k
        reps[f"kmer{k}"] = kmer_matrix(list(accs))
        SA.K = old
    reps["multik"] = np.hstack([reps[f"kmer{k}"] for k in (3, 4, 5, 6)]).astype(np.float32)

    log("  ORFs, códons e dicódons em 6 quadros ...")
    frows, codon, dicod = [], [], []
    t0 = time.time()
    for i, a in enumerate(accs, 1):
        f, c, d = orf_features(seqs[a])
        frows.append(f); codon.append(c); dicod.append(d)
        if i % 300 == 0:
            log(f"    {i}/{len(accs)} ({(time.time()-t0)/i:.2f}s/genoma)")
    orf_df = pd.DataFrame(frows)
    reps["orf"] = orf_df.values.astype(np.float32)
    reps["codon"] = np.array(codon, dtype=np.float32)
    reps["dicodon"] = np.array(dicod, dtype=np.float32)
    reps["gc_len"] = gclen(md).astype(np.float32)
    reps["compo_all"] = np.hstack([reps["multik"], reps["codon"],
                                   reps["orf"]]).astype(np.float32)
    return reps, orf_df.columns.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "json"))
    ap.add_argument("--limit", type=int, default=0, help="dry-run: só os N primeiros genomas.")
    a = ap.parse_args()
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)

    X20, acc20 = load_20b(18, "fp8")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe = pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")
    md = pd.DataFrame(index=accs)
    md["family"] = man.reindex(accs)["family"].replace("", np.nan).values
    for c, _ in PRIMARY:
        md[c] = fe.reindex(accs)[c].values
    md["GC"] = fe.reindex(accs)["GC"].values
    md["genome_length"] = fe.reindex(accs)["genome_length"].values

    keep = md["family"].notna().values
    accs, md, X20 = accs[keep], md[keep], X20.astype(np.float32)[keep]
    if a.limit:
        accs, md, X20 = accs[:a.limit], md.iloc[:a.limit], X20[:a.limit]
    log(f"população: {len(accs)} genomas com família")

    reps, orf_cols = build_reps(accs, md)
    reps["evo2_20b_blocks18"] = X20
    log("dimensionalidades: " + ", ".join(f"{k}={v.shape[1]}" for k, v in sorted(reps.items())))

    seqs = seqs_for(list(accs))
    cu = [dinuc_oe(seqs.get(x, "")) for x in accs]
    md["cpg_oe"] = [c for c, _ in cu]; md["upa_oe"] = [u for _, u in cu]

    fam = md["family"].values.astype(str)
    g95 = cl95_groups(accs)

    out = {"population": {"n": int(len(accs)), "n_families": int(pd.Series(fam).nunique())},
           "config": {"splits": SPLITS, "repeats": REPEATS, "min_orf_bp": MIN_ORF_BP,
                      "kmers": [3, 4, 5, 6], "orf_features": orf_cols,
                      "dims": {k: int(v.shape[1]) for k, v in reps.items()},
                      "coverage_note": "k>6 nao rodado: 16.384+ dims dominariam o custo sem "
                                       "acrescentar ao argumento, ja que dicodon cobre a mesma "
                                       "escala com quadro de leitura."},
           "targets": {}}

    order = ["evo2_20b_blocks18", "kmer3", "kmer4", "kmer5", "kmer6", "multik",
             "codon", "dicodon", "orf", "compo_all", "gc_len"]
    for name, logt in PRIMARY + NEGCTRL:
        y = md[name].values.astype(float)
        m = np.isfinite(y)
        yv = np.log1p(np.clip(y[m], 0, None)) if logt else y[m]
        rec = {"log1p": logt, "n": int(m.sum()), "primary": (name, logt) in PRIMARY, "reps": {}}
        log(f"\n=== {name} (n={m.sum()}) ===")
        for r in order:
            Xr = reps[r][m]
            s95 = reg_group(Xr, yv, g95[m])
            sfam = reg_group(Xr, yv, fam[m])
            rec["reps"][r] = {"cl95": {"mean": float(s95.mean()), "std": float(s95.std()),
                                       "scores": s95.tolist()},
                              "family": {"mean": float(sfam.mean()), "std": float(sfam.std()),
                                         "scores": sfam.tolist()}}
            log(f"  {r:>20}  cl95={s95.mean():+.3f}  family={sfam.mean():+.3f}")
        out["targets"][name] = rec

    # ---- Evo 2 contra CADA baseline, e contra o MELHOR baseline por alvo
    tests = {}
    for scheme in ("cl95", "family"):
        per_target = {}
        for name, _ in PRIMARY:
            e = out["targets"][name]["reps"]["evo2_20b_blocks18"][scheme]["scores"]
            rows, pv, names = {}, [], []
            for r in order:
                if r == "evo2_20b_blocks18":
                    continue
                b = out["targets"][name]["reps"][r][scheme]["scores"]
                t, p = nadeau_bengio(e, b); lo, hi = boot_ci(e, b)
                rows[r] = {"delta": float(np.mean(e) - np.mean(b)), "ci95": [lo, hi],
                           "t": t, "p_raw": p}
                pv.append(p); names.append(r)
            adj = holm(pv, names)
            for r in names:
                rows[r]["p_holm_within_target"] = adj[r]
            entry = {"vs_each": rows, "evo2_r2": float(np.mean(e)), "by_class": {}}
            for cls, members in CLASSES.items():
                ms = [r for r in members if r in rows]
                best = max(ms, key=lambda r: out["targets"][name]["reps"][r][scheme]["mean"])
                entry["by_class"][cls] = {
                    "best": best,
                    "best_r2": out["targets"][name]["reps"][best][scheme]["mean"],
                    "delta": rows[best]["delta"], "ci95": rows[best]["ci95"],
                    "p": rows[best]["p_raw"],
                    "evo2_ahead": bool(rows[best]["delta"] > 0 and rows[best]["p_raw"] < 0.05)}
            per_target[name] = entry
        tests[scheme] = per_target
    out["tests"] = tests
    out["classes"] = CLASSES
    out["classes_note"] = (
        "R1 3.4 pede baselines de COMPOSICAO. Um scan de ORF nao e composicao: e uma versao "
        "simplificada do pipeline de anotacao que produziu os alvos, entao as duas classes sao "
        "reportadas separadas. codon/dicodon entram na classe composicional apesar de exigirem "
        "um scan de 6 quadros para definir quadro de leitura -- escolha conservadora, que "
        "fortalece a classe composicional.")

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "composition_baselines_metrics.json")
    json.dump(out, open(dst, "w"), indent=1)
    log(f"\n-> {dst}")
    for scheme in ("cl95", "family"):
        for cls in CLASSES:
            log(f"\n### {scheme} | Evo 2 vs melhor baseline da classe {cls}")
            for name, v in tests[scheme].items():
                c = v["by_class"][cls]
                mark = "OK " if c["evo2_ahead"] else "NAO"
                log(f"  {mark} {name:<22} evo2={v['evo2_r2']:+.3f} "
                    f"melhor={c['best']}({c['best_r2']:+.3f}) "
                    f"delta={c['delta']:+.3f} p={c['p']:.3g}")


if __name__ == "__main__":
    main()
