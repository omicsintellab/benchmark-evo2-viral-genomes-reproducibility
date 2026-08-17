#!/usr/bin/env python3
"""01_block_shuffle.py — controle de embaralhamento de blocos (revisão R1 3.2/3.3).

R1 3.3: "os experimentos não testam ordem gênica nem arranjo do genoma; o mean-pooling
pode preservar propriedades globais e perder posição". O revisor sugere exatamente este
teste: reordenar blocos grandes do genoma preservando a composição local e comparar os
embeddings do original e do rearranjado.

Desenho:
  - Para cada genoma, gera versões com a ORDEM dos blocos permutada (n_blocks
    configurável). A composição global é IDÊNTICA (mesmos nucleotídeos) e a composição
    local é quase idêntica (só as junções mudam); o que muda é o arranjo de longo alcance.
  - Embeda original E embaralhados NA MESMA CORRIDA, com o mesmo modelo e precisão.
    Isso é essencial: comparar o embaralhado contra o cache FP8 antigo confundiria efeito
    de rearranjo com efeito de precisão/hardware.
  - Uma varredura de n_blocks (poucos blocos = perturbação suave; muitos = severa) dá
    resposta dose-dependente, que é bem mais informativa que um único ponto.

O que sai daqui é o .npz com os embeddings; as métricas (deslocamento do embedding,
transferência do probe) são calculadas em CPU depois, por `code/03_analysis/`.

Interpretação, decidida antes de ver o resultado:
  - Se o embedding quase não se move ao permutar blocos, o Evo 2 NÃO representa arranjo
    de longo alcance e a claim tem de ser estreitada para features-resumo de nível
    genômico — que é o desfecho que R1 3.3 já autoriza.
  - Se o embedding se move muito além do ruído, há representação de arranjo, e aí a
    claim de arquitetura fica sustentada por evidência direta, não por probe indireto.

Uso:
    python 01_block_shuffle.py --model evo2_20b --weights-local /path \\
        --fasta all_genomes.fasta --subset probe_subset_features.tsv \\
        --n-genomes 200 --blocks 4,16 --out-dir ./out
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evo2_compat import log, patch_fp8, read_fasta, mean_pool_embed, resolve_weights, shard_pipeline

LAYER = {"evo2_20b": "blocks.18.mlp.l3", "evo2_7b": "blocks.28.mlp.l3"}


def block_shuffle(seq, n_blocks, rng):
    """Permuta a ordem de n_blocks blocos contíguos. Garante permutação != identidade."""
    L = len(seq)
    if L < n_blocks * 100:
        return None
    edges = [round(i * L / n_blocks) for i in range(n_blocks + 1)]
    blocks = [seq[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    for _ in range(50):
        perm = rng.permutation(n_blocks)
        if not np.array_equal(perm, np.arange(n_blocks)):
            break
    return "".join(blocks[i] for i in perm), perm.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_20b")
    ap.add_argument("--weights-local", default=None,
                    help="Caminho do .pt OU raiz que contenha <model>/<model>.pt.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--subset", required=True, help="TSV com coluna 'accession'.")
    ap.add_argument("--n-genomes", type=int, default=200)
    ap.add_argument("--blocks", default="4,16", help="Lista de n_blocks, separada por vírgula.")
    ap.add_argument("--layer", default=None)
    ap.add_argument("--window", type=int, default=32768)
    ap.add_argument("--stride", type=int, default=16384)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sharded", action="store_true", help="Fatia via accelerate.")
    ap.add_argument("--pipeline", action="store_true",
                    help="Fatiamento manual em pipeline pelos blocos (ver 00_preflight).")
    ap.add_argument("--out-dir", default="./out")
    a = ap.parse_args()

    layer = a.layer or LAYER[a.model]
    blocks = [int(x) for x in a.blocks.split(",") if x.strip()]
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- 1. genomas
    import pandas as pd
    accs = pd.read_csv(a.subset, sep="\t")["accession"].astype(str).tolist()
    seqs = read_fasta(a.fasta, wanted=set(accs))
    log(f"CONTROLE POSITIVO — accessions pedidas: {len(accs)} | achadas no FASTA: {len(seqs)}")
    if len(seqs) < len(accs):
        log(f"  AVISO: {len(accs)-len(seqs)} ausentes; o FASTA pode estar incompleto")
    accs = [x for x in accs if x in seqs]
    if a.n_genomes and a.n_genomes < len(accs):
        idx = rng.choice(len(accs), a.n_genomes, replace=False)
        accs = [accs[i] for i in sorted(idx)]
    log(f"genomas a processar: {len(accs)} | condições: original + {blocks}")

    # ---- 2. modelo
    wl = resolve_weights(a.weights_local, a.model)
    patch_fp8(a.model)
    from evo2 import Evo2
    model = Evo2(a.model, local_path=wl)
    if a.pipeline:
        try: model.model.to("cpu")
        except Exception: pass
        shard_pipeline(model.model)
    elif a.sharded:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        import torch
        inner = model.model
        try: inner.to("cpu")
        except Exception: pass
        ns = [type(inner).__name__]
        mm = get_balanced_memory(inner, dtype=torch.bfloat16, no_split_module_classes=ns)
        dmap = infer_auto_device_map(inner, max_memory=mm, dtype=torch.bfloat16,
                                     no_split_module_classes=ns)
        if any(v in ("cpu", "disk") for v in dmap.values()):
            log("ERRO: device_map jogou módulos p/ CPU/disk — não cabe. Rode 00_preflight.py.")
            sys.exit(2)
        model.model = dispatch_model(inner, device_map=dmap)
        log(f"modelo fatiado em {len(set(dmap.values()))} devices")
    else:
        model.model.to(a.device)
    log(f"modelo {a.model} pronto | camada {layer}")

    # ---- 3. embeddings
    conds = ["original"] + [f"shuf{b}" for b in blocks]
    X = {c: [] for c in conds}
    kept, perms, short, t0 = [], {}, {}, time.time()
    for i, acc in enumerate(accs, 1):
        s = seqs[acc].upper()
        try:
            row = {"original": mean_pool_embed(model, layer, s, a.window, a.stride,
                                               a.max_windows, a.device)}
            # Pular apenas a CONDIÇÃO que não cabe, nunca o genoma inteiro. Descartar o
            # genoma por causa do maior n_blocks do sweep filtra por COMPRIMENTO — e o
            # comprimento é o confundidor central deste experimento, então o corte enviesa
            # a amostra para genomas longos justamente onde isso mais importa. Medido: com
            # o sweep incluindo 32 blocos, o descarte tirava 61 de 200 genomas e SUBESTIMAVA
            # o deslocamento (1,8-3,3% contra 3,4-6,8% na amostra sem viés).
            for b in blocks:
                r = block_shuffle(s, b, rng)
                if r is None:
                    short[f"shuf{b}"] = short.get(f"shuf{b}", 0) + 1
                    continue
                sh, perm = r
                perms.setdefault(acc, {})[str(b)] = perm
                row[f"shuf{b}"] = mean_pool_embed(model, layer, sh, a.window, a.stride,
                                                 a.max_windows, a.device)
            for c in conds:
                X[c].append(row.get(c))     # None onde a condição não coube
            kept.append(acc)
        except Exception as e:
            log(f"  falhou {acc}: {type(e).__name__}: {e}")
            continue
        if i % 10 == 0:
            el = time.time() - t0
            log(f"  {i}/{len(accs)} | {el/i:.1f}s/genoma | ETA {(len(accs)-i)*el/i/60:.0f} min")

    if not kept:
        log("nada processado — abortando"); sys.exit(1)

    # Cada condição pode cobrir um subconjunto diferente (genoma curto demais para aquele
    # n_blocks). Grava-se, por condição embaralhada, a matriz E os índices das linhas de
    # `original` a que ela corresponde — assim o pareamento fica explícito no arquivo, em vez
    # de depender de as matrizes terem o mesmo número de linhas.
    dst = os.path.join(a.out_dir, f"block_shuffle_{a.model}_L{layer.split('.')[1]}_w{a.window}.npz")
    arrays = {"accs": np.array(kept),
              "original": np.array(X["original"], dtype=np.float32)}
    coverage = {}
    for c in conds:
        if c == "original":
            continue
        idx = [i for i, v in enumerate(X[c]) if v is not None]
        arrays[c] = np.array([X[c][i] for i in idx], dtype=np.float32)
        arrays[f"{c}_idx"] = np.array(idx, dtype=np.int32)
        coverage[c] = len(idx)
    np.savez_compressed(dst, **arrays)

    meta = {"model": a.model, "layer": layer, "window": a.window, "stride": a.stride,
            "max_windows": a.max_windows, "seed": a.seed, "blocks": blocks,
            "n_genomes": len(kept), "conditions": conds, "permutations": perms,
            "coverage_per_condition": coverage,
            "skipped_too_short_per_condition": short,
            "sharded": a.sharded, "elapsed_min": round((time.time() - t0) / 60, 1)}
    json.dump(meta, open(os.path.join(a.out_dir, "block_shuffle_meta.json"), "w"), indent=1)
    log(f"cobertura por condição (de {len(kept)} genomas):")
    for c, n in coverage.items():
        miss = short.get(c, 0)
        log(f"  {c:>8}: {n:4d}" + (f"   ({miss} curtos demais para esse n_blocks)" if miss else ""))
    if short:
        log("  NOTA: condições com cobertura menor sao enviesadas para genomas LONGOS; "
            "comparar dose-resposta entre condicoes so no subconjunto comum.")
    log(f"-> {dst}  ({len(kept)} genomas x {len(conds)} condições, "
        f"{meta['elapsed_min']} min)")


if __name__ == "__main__":
    main()
