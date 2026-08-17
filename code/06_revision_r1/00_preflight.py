#!/usr/bin/env python3
"""00_preflight.py — decide o que roda nesta GPU antes de gastar hora (revisão R1).

RODAR ISTO PRIMEIRO. O 20B em bf16 pesa ~40 GB e os scripts do artigo são single-GPU
(`--device cuda:0`), o que funcionou porque a corrida original usou UM H100 de 80 GB.
Em qualquer outra máquina a primeira pergunta é: cabe? E até que comprimento de contexto?

Referência medida:
  4x L4  (g6.12xlarge,  ~22 GB/card): 20B NÃO cabe em card nenhum. 7B vai só até 8k.
  4x L40S (g6e.12xlarge, ~48 GB/card): 40 GB de pesos + ativação — o single-GPU volta a ser
    plausível, mas fica APERTADO em 32k (a doc de infra do projeto já marcava isso). Se A
    passar em 32k, é o caminho: nada de fatiamento, idêntico ao desenho do artigo.

Cada estratégia roda em **processo separado**. Isso não é zelo: um carregamento que falha
por OOM deixa o modelo parcial referenciado no traceback, a VRAM não volta com
`gc.collect()`/`empty_cache()`, e a estratégia seguinte transborda para outra GPU — foi
exatamente assim que o teste do 7B falhou com "two devices, cuda:0 and cuda:1" numa versão
anterior deste script. Subprocesso é a única garantia de VRAM limpa.

As estratégias single-GPU rodam com CUDA_VISIBLE_DEVICES=0, para que um "sucesso" não seja
na verdade um vazamento silencioso para os outros cards.

Estratégias:
  A) 20B single-GPU           — FALHA no L4 (40 GB de pesos em 22 GB úteis). Confirmado.
  B) 20B via accelerate       — CARREGA, mas o forward morre com device misto: o
                                dispatch_model pressupõe que todo dado passa pelas
                                assinaturas de forward, e o StripedHyena não satisfaz isso.
  C) 7B single-GPU            — funciona até 8k de contexto; OOM em 16k.
  D) 7B via accelerate        — mesma dúvida de B.
  E) 20B pipeline manual      — fatiamento explícito pelos blocos (evo2_compat.shard_pipeline).
  F) 7B pipeline manual       — idem, para chegar a 32k no 7B.

Uso:
    python 00_preflight.py --weights-local /mnt/.../weights --out out/preflight_report.json
    python 00_preflight.py --only C --weights-local ...     # uma estratégia só
"""
import os, sys, json, time, argparse, subprocess, traceback

STRATEGIES = {"A": ("A_20b_single", "evo2_20b", "single"),
              "B": ("B_20b_sharded", "evo2_20b", "sharded"),
              "C": ("C_7b_single", "evo2_7b", "single"),
              "D": ("D_7b_sharded", "evo2_7b", "sharded"),
              "E": ("E_20b_pipeline", "evo2_20b", "pipeline"),
              "F": ("F_7b_pipeline", "evo2_7b", "pipeline")}
# D existe porque o 7B num L4 só sozinho satura em 8k de contexto (pico 16,8 GB de 22),
# e o artigo usa janela de 32k. Fatiado, os pesos ocupam ~3,5 GB por card e sobra muito
# mais espaço para ativação — é o caminho mais provável para chegar a 32k nesta máquina.


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def gpu_inventory():
    import torch
    inv = {"cuda_available": torch.cuda.is_available(),
           "torch": torch.__version__, "n_gpu": 0, "gpus": []}
    if not inv["cuda_available"]:
        return inv
    inv["n_gpu"] = torch.cuda.device_count()
    for i in range(inv["n_gpu"]):
        p = torch.cuda.get_device_properties(i)
        inv["gpus"].append({"index": i, "name": p.name,
                            "capability": f"{p.major}.{p.minor}",
                            "total_gb": round(p.total_memory / 1024**3, 1)})
    inv["total_vram_gb"] = round(sum(g["total_gb"] for g in inv["gpus"]), 1)
    inv["fp8_capable"] = all(tuple(map(int, g["capability"].split("."))) >= (8, 9)
                             for g in inv["gpus"])
    return inv


def ctx_probe(model, device, lengths=(4096, 8192, 16384, 32768)):
    """Pico de memória por comprimento de contexto. Para no primeiro erro."""
    import torch
    out = {}
    seq = "ACGT" * (max(lengths) // 4 + 1)
    for L in lengths:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        try:
            ids = torch.tensor(model.tokenizer.tokenize(seq[:L]),
                               dtype=torch.int).unsqueeze(0).to(device)
            with torch.inference_mode():
                model.model.forward(ids)
            peak = max(torch.cuda.max_memory_allocated(i)
                       for i in range(torch.cuda.device_count()))
            out[str(L)] = {"ok": True, "peak_gb": round(peak / 1024**3, 2)}
            log(f"    contexto {L}: OK, pico {out[str(L)]['peak_gb']} GB")
            del ids
        except Exception as e:
            out[str(L)] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
            log(f"    contexto {L}: FALHOU — {type(e).__name__}")
            torch.cuda.empty_cache()
            break
        torch.cuda.empty_cache()
    return out


def run_one(model_name, mode, weights_local, use_fp8):
    """Executado DENTRO do subprocesso. Devolve o dict de resultado."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evo2_compat import patch_fp8, resolve_weights, shard_pipeline, describe_devices
    wl = resolve_weights(weights_local, model_name)
    fp8 = patch_fp8(model_name) if use_fp8 else None
    if not use_fp8:
        import yaml, evo2.models as M
        res = M.CONFIG_MAP[model_name]
        cfg = yaml.safe_load(M.pkgutil.get_data(M.__name__, res))
        cfg["use_fp8_input_projections"] = False
        patched = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
        if not hasattr(M, "_orig_get_data"):
            M._orig_get_data = M.pkgutil.get_data
        _o = M._orig_get_data
        M.pkgutil.get_data = lambda p, r: patched if (p == M.__name__ and r == res) else _o(p, r)
        log("FP8 DESLIGADO por --no-fp8 (embeddings NÃO comparáveis aos cacheados)")
    from evo2 import Evo2

    t0 = time.time()
    if mode == "single":
        m = Evo2(model_name, local_path=wl)
        m.model.to("cuda:0")
        dev = "cuda:0"
        extra = {}
    elif mode == "pipeline":
        m = Evo2(model_name, local_path=wl)
        try:
            m.model.to("cpu")
        except Exception:
            pass
        extra = {"pipeline": shard_pipeline(m.model)}
        dev = "cuda:0"
    else:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        m = Evo2(model_name, local_path=wl)
        inner = m.model
        try:
            inner.to("cpu")
        except Exception:
            pass
        ns = [type(inner).__name__]
        mm = get_balanced_memory(inner, dtype=torch.bfloat16, no_split_module_classes=ns)
        dmap = infer_auto_device_map(inner, max_memory=mm, dtype=torch.bfloat16,
                                     no_split_module_classes=ns)
        n_off = sum(1 for v in dmap.values() if v in ("cpu", "disk"))
        log(f"    device_map: {len(set(dmap.values()))} devices, {n_off} módulos fora da GPU")
        if n_off:
            raise RuntimeError(f"device_map jogou {n_off} módulos para CPU/disk — não cabe")
        m.model = dispatch_model(inner, device_map=dmap)
        dev = "cuda:0"
        extra = {"n_devices": len(set(map(str, dmap.values())))}
    load_s = round(time.time() - t0)
    log(f"  carregou em {load_s}s — sondando contexto")
    ctx = ctx_probe(m, dev)
    res = {"loaded": True, "load_seconds": load_s, "fp8": fp8, "context": ctx, **extra}
    # se falhou por device misto, o mapa de devices é o que permite consertar
    if any((not v.get("ok")) and "same device" in v.get("error", "") for v in ctx.values()):
        res["device_map_dump"] = describe_devices(m.model)
        log("  device misto — mapa por módulo:", json.dumps(res["device_map_dump"], indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-local", default=None,
                    help="Caminho do .pt OU raiz que contenha <model>/<model>.pt.")
    ap.add_argument("--out", default="preflight_report.json")
    ap.add_argument("--only", default=None, choices=list(STRATEGIES),
                    help="Roda uma estratégia só (usado internamente pelos subprocessos).")
    ap.add_argument("--no-fp8", action="store_true",
                    help="Desliga FP8. Só para diagnóstico — bf16 muda os embeddings (Tabela S1).")
    ap.add_argument("--child-out", default=None, help="Uso interno.")
    a = ap.parse_args()

    # ---- modo filho: uma estratégia, um processo
    if a.only:
        label, model_name, mode = STRATEGIES[a.only]
        log(f"--- estratégia {label} ({model_name}, {mode}) ---")
        try:
            res = run_one(model_name, mode, a.weights_local, not a.no_fp8)
        except Exception as e:
            res = {"loaded": False, "error": f"{type(e).__name__}: {e}"[:600],
                   "traceback": traceback.format_exc()[-2000:]}
            log(f"  FALHOU: {type(e).__name__}: {e}")
        json.dump(res, open(a.child_out, "w"), indent=1)
        return

    # ---- modo pai: inventário + um subprocesso por estratégia
    rep = {"inventory": gpu_inventory(), "strategies": {}}
    log("inventário:", json.dumps(rep["inventory"], indent=1))
    if not rep["inventory"]["cuda_available"]:
        log("sem CUDA — abortando"); json.dump(rep, open(a.out, "w"), indent=1); return

    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    for key, (label, model_name, mode) in STRATEGIES.items():
        child = os.path.join(outdir, f".preflight_{key}.json")
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if mode == "single":
            # trava em 1 GPU: um "sucesso" não pode ser vazamento para os outros cards
            env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [sys.executable, os.path.abspath(__file__), "--only", key,
               "--child-out", child]
        if a.weights_local:
            cmd += ["--weights-local", a.weights_local]
        if a.no_fp8:
            cmd.append("--no-fp8")
        log(f"=== subprocesso {label} "
            f"(CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'todas')}) ===")
        p = subprocess.run(cmd, env=env)
        if os.path.exists(child):
            rep["strategies"][label] = json.load(open(child))
            os.remove(child)
        else:
            rep["strategies"][label] = {"loaded": False,
                                        "error": f"subprocesso morreu (exit {p.returncode}) "
                                                 f"sem escrever resultado — provável OOM do "
                                                 f"kernel ou crash nativo"}

    ok = {L: [k for k, v in rep["strategies"].items()
              if v.get("loaded") and v.get("context", {}).get(str(L), {}).get("ok")]
          for L in (4096, 32768)}
    rep["verdict"] = {
        "usable_at_4k": ok[4096], "usable_at_32k": ok[32768],
        "recommended": ok[32768][0] if ok[32768] else None,
        "note": ("Rodar 01/02 com --model conforme 'recommended'. Se só C_7b_single passar, "
                 "as claims de arquitetura ficam demonstradas no 7B e o 20B espera hardware "
                 "com >= 80 GB numa GPU só. Se B falhar por falta de accelerate, "
                 "`pip install accelerate` e rodar de novo.")}
    log("VEREDITO:", json.dumps(rep["verdict"], indent=1))
    json.dump(rep, open(a.out, "w"), indent=1)
    log(f"-> {a.out}")


if __name__ == "__main__":
    main()
