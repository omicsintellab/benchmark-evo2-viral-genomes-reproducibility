#!/usr/bin/env python3
"""02_generation_rerun.py — geração re-rodada PERSISTINDO as sequências (revisão R1 4.2, R2 #11/#12).

R2 #12: "a similaridade de 4-mer mede sobretudo composição e não avalia recuperação da
sequência correta nem preservação da estrutura codificante. Devem ser incluídas identidade
de sequência, distância de edição, continuidade de ORF, frequência de frameshift e stop
prematuro. Resultados por genoma, estimativas de incerteza e sensibilidade aos parâmetros
de decodificação também devem ser apresentados."

Nada disso é calculável a partir do que temos: o script original consumia as sequências
geradas em memória e gravava só bits/nt e cosseno de 4-mer. Este script re-roda a geração
**salvando as sequências**, com replicatas por genoma e varredura de temperatura/top_k.
As métricas caras de comparar sequência (identidade, edit distance, ORF, stop prematuro)
saem depois em CPU — aqui só se produz o material.

Mantém as MESMAS accessions e os mesmos prompt_bp/gap_bp da corrida original, lidos do CSV
já publicado, para que a comparação com o resultado do artigo seja limpa.

Uso:
    python 02_generation_rerun.py --model evo2_20b --weights-local /path \\
        --fasta all_genomes.fasta --ref-csv generation_completion_evo2_20b.csv \\
        --temperatures 0.7,1.0 --top-ks 4 --n-samples 3 --out-dir ./out
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evo2_compat import log, patch_fp8, read_fasta, resolve_weights, shard_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_20b")
    ap.add_argument("--weights-local", default=None,
                    help="Caminho do .pt OU raiz que contenha <model>/<model>.pt.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ref-csv", required=True,
                    help="generation_completion_evo2_20b.csv da corrida original "
                         "(define accessions e set_label).")
    ap.add_argument("--prompt-bp", type=int, default=4096)
    ap.add_argument("--gap-bp", type=int, default=1024)
    ap.add_argument("--temperatures", default="0.7,1.0")
    ap.add_argument("--top-ks", default="4")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="Replicatas por genoma e por configuração (dá a incerteza).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="dry-run: só os N primeiros.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Prompts por chamada de generate(). Batch 1 desperdiça a GPU; "
                         "em OOM o lote se divide sozinho.")
    ap.add_argument("--sharded", action="store_true", help="Fatia via accelerate.")
    ap.add_argument("--pipeline", action="store_true",
                    help="Fatiamento manual em pipeline pelos blocos (ver 00_preflight).")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Indice deste processo (0-based) ao dividir o trabalho.")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="Quantos processos dividem o trabalho. Com o modelo fatiado em 2 "
                         "GPUs, 2 processos usam as 4 de fato e dobram a vazao.")
    ap.add_argument("--out-dir", default="./out")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    sfx = f"_s{a.shard_index}" if a.shard_count > 1 else ""
    temps = [float(x) for x in a.temperatures.split(",") if x.strip()]
    topks = [int(x) for x in a.top_ks.split(",") if x.strip()]

    ref = pd.read_csv(a.ref_csv)
    need = a.prompt_bp + a.gap_bp
    seqs = read_fasta(a.fasta, wanted=set(ref["accession"].astype(str)))
    log(f"CONTROLE POSITIVO — accessions no CSV: {len(ref)} | achadas no FASTA: {len(seqs)}")
    ref = ref[ref["accession"].astype(str).isin(seqs)].reset_index(drop=True)
    ref = ref[[len(seqs[x]) >= need for x in ref["accession"].astype(str)]].reset_index(drop=True)
    if a.limit:
        ref = ref.head(a.limit)
    if a.shard_count > 1:
        ref = ref.iloc[a.shard_index::a.shard_count].reset_index(drop=True)
        log(f"shard {a.shard_index+1}/{a.shard_count}: {len(ref)} genomas deste processo")
    log(f"genomas: {len(ref)} | temps {temps} | top_k {topks} | {a.n_samples} replicatas")
    log(f"total de gerações: {len(ref)*len(temps)*len(topks)*a.n_samples}")

    wl = resolve_weights(a.weights_local, a.model)
    patch_fp8(a.model)
    from evo2 import Evo2
    import torch
    model = Evo2(a.model, local_path=wl)
    if a.pipeline:
        try: model.model.to("cpu")
        except Exception: pass
        shard_pipeline(model.model)
    elif a.sharded:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        inner = model.model
        try: inner.to("cpu")
        except Exception: pass
        ns = [type(inner).__name__]
        mm = get_balanced_memory(inner, dtype=torch.bfloat16, no_split_module_classes=ns)
        dmap = infer_auto_device_map(inner, max_memory=mm, dtype=torch.bfloat16,
                                     no_split_module_classes=ns)
        if any(v in ("cpu", "disk") for v in dmap.values()):
            log("ERRO: não cabe nas GPUs. Rode 00_preflight.py."); sys.exit(2)
        model.model = dispatch_model(inner, device_map=dmap)
    else:
        model.model.to(a.device)
    log("modelo pronto")

    import gc

    # Nomes sob os quais o Evo2/vortex costuma pendurar o cache de inferência. O log
    # "Initializing inference params with max_seqlen=..." aparece a CADA generate(), e o
    # cache anterior fica preso ao objeto do modelo: a GPU 0 chegou a 39,5 GiB quando os
    # pesos dela são ~12 GiB. Sem limpar, a corrida estoura no meio mesmo com batch 1.
    STATE_ATTRS = ["inference_params", "_inference_params", "inference_params_dict",
                   "cache", "_cache", "kv_cache"]

    def free_generation_state():
        for obj in (model, getattr(model, "model", None)):
            if obj is None:
                continue
            for attr in STATE_ATTRS:
                if hasattr(obj, attr):
                    try:
                        setattr(obj, attr, None)
                    except Exception:
                        pass
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    def _gen_once(prefixes, n, temperature, top_k):
        out = model.generate(prompt_seqs=list(prefixes), n_tokens=n,
                             temperature=temperature, top_k=top_k)
        g = getattr(out, "sequences", None)
        if g is None:
            g = out["sequences"] if isinstance(out, dict) else out
        if len(g) != len(prefixes):
            raise RuntimeError(f"generate devolveu {len(g)} para {len(prefixes)} prompts")
        return list(g)

    def gen_batch(prefixes, n, temperature, top_k):
        """Gera um LOTE de prompts. Em falha, divide o lote pela metade até 1.

        A limpeza acontece FORA do `except`: dentro dele o traceback mantém referência aos
        tensores da tentativa que estourou, então `empty_cache()` não devolve nada e a
        retentativa nasce condenada — foi o que aconteceu na primeira versão, em que até o
        lote de 1 falhava depois de um OOM anterior."""
        err = None
        try:
            return _gen_once(prefixes, n, temperature, top_k)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"
            oom = "out of memory" in str(e).lower() or "OutOfMemory" in type(e).__name__
        # aqui o except já terminou: o traceback foi descartado e a memória pode voltar
        free_generation_state()
        if len(prefixes) == 1:
            raise RuntimeError(f"geração falhou mesmo com lote 1 — {err}")
        log(f"    lote de {len(prefixes)} falhou ({err[:60]}); dividindo")
        h = len(prefixes) // 2
        return (gen_batch(prefixes[:h], n, temperature, top_k) +
                gen_batch(prefixes[h:], n, temperature, top_k))

    # lista de trabalho: (accession, set_label, T, K, replicata). Agrupada por (T,K) porque
    # generate() recebe uma temperatura/top_k por chamada.
    work = [(str(r.accession), r.set_label, T, K, rep)
            for r in ref.itertuples() for T in temps for K in topks
            for rep in range(a.n_samples)]
    log(f"lista de trabalho: {len(work)} gerações | batch {a.batch_size}")

    fa = open(os.path.join(a.out_dir, f"generated_{a.model}{sfx}.fasta"), "w")
    rows, t0, done = [], time.time(), 0
    torch.manual_seed(a.seed)
    for T in temps:
        for K in topks:
            grp = [w for w in work if w[2] == T and w[3] == K]
            for s in range(0, len(grp), a.batch_size):
                chunk = grp[s:s + a.batch_size]
                prefixes = [seqs[acc].upper()[:a.prompt_bp] for acc, _, _, _, _ in chunk]
                try:
                    gs = gen_batch(prefixes, a.gap_bp, T, K)
                except Exception as e:
                    log(f"  lote falhou de vez (T={T} K={K}): {type(e).__name__}: {e}")
                    continue
                for (acc, label, _, _, rep), g in zip(chunk, gs):
                    sid = f"{acc}|T{T}|K{K}|r{rep}"
                    fa.write(f">{sid}\n{g}\n")
                    rows.append({"accession": acc, "set_label": label,
                                 "temperature": T, "top_k": K, "replicate": rep,
                                 "seq_id": sid, "gen_len": len(g),
                                 "prompt_bp": a.prompt_bp, "gap_bp": a.gap_bp})
                done += len(chunk)
                free_generation_state()   # sem isto o cache de inferência acumula e estoura
                el = time.time() - t0
                free_gb = (min(torch.cuda.mem_get_info(i)[0]
                               for i in range(torch.cuda.device_count())) / 1024**3
                           if torch.cuda.is_available() else float("nan"))
                log(f"  {done}/{len(work)} | {el/done:.1f}s/geração | "
                    f"ETA {(len(work)-done)*el/done/60:.0f} min | "
                    f"GPU mais cheia: {free_gb:.1f} GB livres")
    fa.close()

    # a sequência VERDADEIRA do gap vai junto — sem ela nada se compara depois em CPU
    with open(os.path.join(a.out_dir, f"true_gaps_{a.model}{sfx}.fasta"), "w") as th:
        for r in ref.itertuples():
            acc = str(r.accession); full = seqs[acc].upper()
            th.write(f">{acc}\n{full[a.prompt_bp:a.prompt_bp+a.gap_bp]}\n")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, f"generation_index_{a.model}{sfx}.csv"), index=False)
    meta = {"model": a.model, "n_genomes": int(len(ref)), "temperatures": temps,
            "top_ks": topks, "n_samples": a.n_samples, "prompt_bp": a.prompt_bp,
            "gap_bp": a.gap_bp, "seed": a.seed, "n_generated": int(len(df)),
            "sharded": a.sharded, "elapsed_min": round((time.time() - t0) / 60, 1)}
    json.dump(meta, open(os.path.join(a.out_dir, f"generation_rerun_meta{sfx}.json"), "w"), indent=1)
    log(f"-> {len(df)} sequências geradas em {meta['elapsed_min']} min, em {a.out_dir}")


if __name__ == "__main__":
    main()
