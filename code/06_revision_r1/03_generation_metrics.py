#!/usr/bin/env python3
"""03_generation_metrics.py — métricas das sequências geradas (revisão R1 4.2, R2 #12).

Roda em CPU, sobre a saída de `02_generation_rerun.py`. Faz duas coisas:

1) **CONTROLE POSITIVO do fatiamento.** O CSV da corrida original traz `kmer_evo` (cosseno de
   4-mer contra o gap verdadeiro) das MESMAS accessions, medido num H100 em FP8 e sem
   fatiamento. Se o modelo fatiado produzir valores na mesma faixa, é evidência de que o
   pipeline não corrompeu o modelo. Se produzir sequência degenerada — homopolímero, um único
   nucleotídeo, composição fora da faixa — o cosseno denuncia antes de se gastar horas.
   **Rodar isto no dry-run, antes da corrida cheia.**

   O controle é calculado APENAS na célula (temperature, top_k) que a corrida original usou
   (`--ref-temperature`, `--ref-top-k`; padrão 0,7 e 4). As demais células da varredura são
   reportadas à parte, rotuladas como efeito da decodificação — não são controle, porque a
   corrida original nunca as rodou.

2) **As métricas que R2 #12 exige** e que não eram calculáveis com o que o script original
   guardava: identidade de sequência, distância de edição, continuidade de ORF, frequência de
   frameshift e stop prematuro — por genoma, com incerteza entre replicatas e por configuração
   de decodificação.

Uso:
    python 03_generation_metrics.py --gen out/generated_evo2_20b.fasta \\
        --true out/true_gaps_evo2_20b.fasta --index out/generation_index_evo2_20b.csv \\
        --ref inputs/generation_completion_evo2_20b.csv --out out/generation_metrics.json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evo2_compat import log, read_fasta

STOP_CODONS = {"TAA", "TAG", "TGA"}


def kmer_vec(s, k=4):
    s = s.upper()
    d = {}
    for i in range(len(s) - k + 1):
        w = s[i:i + k]
        if set(w) <= set("ACGT"):
            d[w] = d.get(w, 0) + 1
    if not d:
        return {}
    tot = sum(d.values())
    return {w: c / tot for w, c in d.items()}


def cosine(a, b):
    if not a or not b:
        return float("nan")
    keys = set(a) | set(b)
    va = np.array([a.get(w, 0.0) for w in keys])
    vb = np.array([b.get(w, 0.0) for w in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na and nb else float("nan")


def max_homopolymer(s):
    best = run = 1
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] else 1
        best = max(best, run)
    return best


def edit_distance(a, b):
    """edlib se disponível (rápido); senão Biopython; senão NaN, declarado."""
    try:
        import edlib
        return float(edlib.align(a, b, task="distance")["editDistance"]), "edlib"
    except ImportError:
        pass
    try:
        from Bio import Align
        al = Align.PairwiseAligner(mode="global", match_score=0, mismatch_score=-1,
                                   open_gap_score=-1, extend_gap_score=-1)
        return float(-al.score(a, b)), "biopython"
    except Exception:
        return float("nan"), "indisponivel"


def orf_stats(s):
    """Continuidade de ORF e stops prematuros no quadro 0 (o quadro em que a geração
    continua o prompt), mais o maior ORF em qualquer quadro."""
    s = s.upper()
    cods = [s[i:i + 3] for i in range(0, len(s) - 2, 3)]
    stops_f0 = sum(1 for c in cods if c in STOP_CODONS)
    first_stop = next((i for i, c in enumerate(cods) if c in STOP_CODONS), None)
    longest = 0
    for f in range(3):
        cl = [s[i:i + 3] for i in range(f, len(s) - 2, 3)]
        run = 0
        for c in cl:
            if c in STOP_CODONS:
                longest = max(longest, run); run = 0
            else:
                run += 1
        longest = max(longest, run)
    return {"stops_frame0": stops_f0,
            "stop_density_frame0": stops_f0 / max(len(cods), 1),
            "codons_to_first_stop": first_stop if first_stop is not None else len(cods),
            "longest_stopfree_run_codons": longest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ref", default=None, help="CSV da corrida original (controle positivo).")
    ap.add_argument("--ref-temperature", type=float, default=0.7,
                    help="Temperatura da corrida original. O controle positivo SÓ é válido "
                         "nesta célula da varredura.")
    ap.add_argument("--ref-top-k", type=int, default=4,
                    help="top_k da corrida original (padrão do generation_completion_20b.py).")
    ap.add_argument("--out", default="generation_metrics.json")
    ap.add_argument("--csv-out", default=None)
    a = ap.parse_args()

    gen = read_fasta(a.gen)
    true = read_fasta(a.true)
    idx = pd.read_csv(a.index)
    log(f"gerações: {len(gen)} | gaps verdadeiros: {len(true)} | linhas no índice: {len(idx)}")
    miss = set(idx["seq_id"]) - set(gen)
    if miss:
        log(f"AVISO: {len(miss)} seq_id do índice sem sequência no FASTA")

    kv_true = {acc: kmer_vec(s) for acc, s in true.items()}
    ed_engine = None
    rows = []
    for r in idx.itertuples():
        g = gen.get(r.seq_id)
        t = true.get(str(r.accession))
        if g is None or t is None:
            continue
        comp = {b: g.upper().count(b) / max(len(g), 1) for b in "ACGT"}
        d, eng = edit_distance(g.upper(), t.upper())
        ed_engine = ed_engine or eng
        o_gen, o_true = orf_stats(g), orf_stats(t)
        rows.append({
            "accession": r.accession, "set_label": r.set_label,
            "temperature": r.temperature, "top_k": r.top_k, "replicate": r.replicate,
            "gen_len": len(g), "true_len": len(t),
            "gc": (comp["G"] + comp["C"]),
            "frac_acgt": sum(comp.values()),
            "max_homopolymer": max_homopolymer(g.upper()),
            "kmer4_cosine": cosine(kmer_vec(g), kv_true.get(str(r.accession), {})),
            "edit_distance": d,
            "identity": 1.0 - d / max(len(t), 1) if np.isfinite(d) else float("nan"),
            "stop_density_frame0_gen": o_gen["stop_density_frame0"],
            "stop_density_frame0_true": o_true["stop_density_frame0"],
            "codons_to_first_stop_gen": o_gen["codons_to_first_stop"],
            "codons_to_first_stop_true": o_true["codons_to_first_stop"],
            "longest_orf_codons_gen": o_gen["longest_stopfree_run_codons"],
            "longest_orf_codons_true": o_true["longest_stopfree_run_codons"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        log("nenhuma sequência pareada — abortando"); sys.exit(1)
    log(f"motor de distância de edição: {ed_engine}")

    out = {"n_sequences": int(len(df)), "edit_distance_engine": ed_engine,
           "by_config": {}, "sanity": {}}

    # ---- 1. controle positivo contra a corrida original
    #
    # O controle SÓ é válido na célula de decodificação que a corrida original usou. Comparar
    # a média sobre toda a varredura contra um CSV gerado com um único (temperature, top_k)
    # mede a varredura, não o pipeline: as células que a original nunca rodou entram no
    # denominador e derrubam a média. Na prática isso já produziu um "delta -0,134, NÃO rodar
    # a corrida cheia" numa corrida que estava correta — o delta na célula pareada era -0,023.
    if a.ref and os.path.exists(a.ref):
        ref = pd.read_csv(a.ref)[["accession", "kmer_evo", "gc_evo"]]
        cell = df[(df["temperature"] == a.ref_temperature) & (df["top_k"] == a.ref_top_k)]
        m = cell.merge(ref, on="accession", how="inner")
        pc = {"ref_temperature": float(a.ref_temperature), "ref_top_k": int(a.ref_top_k),
              "n_matched": int(len(m))}
        if len(m):
            pc.update({
                "kmer4_cosine_new_mean": float(m["kmer4_cosine"].mean()),
                "kmer4_cosine_original_mean": float(m["kmer_evo"].mean()),
                "delta": float(m["kmer4_cosine"].mean() - m["kmer_evo"].mean()),
                "pearson_r": float(m["kmer4_cosine"].corr(m["kmer_evo"]))
                if len(m) > 2 else float("nan"),
            })
        else:
            pc["error"] = ("nenhuma geração na configuração da corrida original; "
                           "ajuste --ref-temperature/--ref-top-k")
        out["sanity"]["positive_control"] = pc

        # as demais células entram como CONTEXTO, explicitamente não comparáveis ao CSV
        others = {}
        for (T, K), g in df.groupby(["temperature", "top_k"]):
            if (T, K) == (a.ref_temperature, a.ref_top_k):
                continue
            gm = g.merge(ref, on="accession", how="inner")
            if len(gm):
                others[f"T{T}_K{K}"] = {
                    "n": int(len(gm)),
                    "kmer4_cosine_new_mean": float(gm["kmer4_cosine"].mean()),
                    "delta_vs_original_run": float(
                        gm["kmer4_cosine"].mean() - gm["kmer_evo"].mean()),
                }
        out["sanity"]["other_configs_not_a_control"] = {
            "note": ("A corrida original usou uma única configuração de decodificação. Estes "
                     "deltas medem o EFEITO DA DECODIFICAÇÃO, não a integridade do pipeline, "
                     "e não devem ser lidos como controle positivo."),
            "by_config": others,
        }
    out["sanity"]["degenerate_check"] = {
        "median_max_homopolymer": float(df["max_homopolymer"].median()),
        "max_max_homopolymer": int(df["max_homopolymer"].max()),
        "min_frac_acgt": float(df["frac_acgt"].min()),
        "median_gc": float(df["gc"].median()),
        "n_len_mismatch": int((df["gen_len"] != df["true_len"]).sum()),
    }

    # ---- 2. métricas por configuração de decodificação e por grupo
    for (T, K), g in df.groupby(["temperature", "top_k"]):
        key = f"T{T}_K{K}"
        out["by_config"][key] = {}
        for label, gg in g.groupby("set_label"):
            out["by_config"][key][str(label)] = {
                m: {"mean": float(gg[m].mean()), "sd": float(gg[m].std()),
                    "n": int(gg[m].notna().sum())}
                for m in ["kmer4_cosine", "identity", "edit_distance", "gc",
                          "stop_density_frame0_gen", "longest_orf_codons_gen"]}
            out["by_config"][key][str(label)]["true_reference"] = {
                m: float(gg[m].mean()) for m in
                ["stop_density_frame0_true", "longest_orf_codons_true"]}

    json.dump(out, open(a.out, "w"), indent=1)
    df.to_csv(a.csv_out or a.out.replace(".json", "_per_sequence.csv"), index=False)

    log("\n=== SANIDADE ===")
    pc = out["sanity"].get("positive_control")
    if pc and "error" in pc:
        log(f"  CONTROLE POSITIVO INDISPONIVEL: {pc['error']}")
    elif pc:
        log(f"  CONTROLE POSITIVO na celula da corrida original "
            f"(T={pc['ref_temperature']}, top_k={pc['ref_top_k']}, n={pc['n_matched']}): "
            f"cosseno 4-mer novo={pc['kmer4_cosine_new_mean']:.3f} vs "
            f"original={pc['kmer4_cosine_original_mean']:.3f} "
            f"(delta {pc['delta']:+.3f}, r={pc['pearson_r']:.3f})")
        log("  -> delta pequeno NESTA celula = o pipeline nao corrompeu o modelo.")
        log("  -> delta grande NESTA celula = investigar antes da corrida cheia.")
    oc = out["sanity"].get("other_configs_not_a_control", {}).get("by_config", {})
    if oc:
        log("  demais configuracoes (efeito da DECODIFICACAO, nao controle):")
        for k, v in sorted(oc.items()):
            log(f"      {k}: cosseno={v['kmer4_cosine_new_mean']:.3f} "
                f"(delta {v['delta_vs_original_run']:+.3f} vs corrida original)")
    dc = out["sanity"]["degenerate_check"]
    log(f"  homopolimero mediano={dc['median_max_homopolymer']:.0f} max={dc['max_max_homopolymer']} "
        f"| GC mediano={dc['median_gc']:.3f} | frac ACGT min={dc['min_frac_acgt']:.3f} "
        f"| comprimentos divergentes={dc['n_len_mismatch']}")
    log(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
