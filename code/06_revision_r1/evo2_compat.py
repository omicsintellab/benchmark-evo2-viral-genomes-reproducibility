#!/usr/bin/env python3
"""evo2_compat.py — helpers compartilhados pelos scripts da revisão R1.

Cópia fiel de `patch_fp8` e `make_windows` de `code/02_embeddings/probe_evo2_viral.py`,
para que os scripts da revisão rodem soltos no Studio sem clonar o repositório inteiro.
Manter em sincronia com o original — a janela/stride/max_windows definem a comparabilidade
com os embeddings já cacheados.
"""
import os
import time
import numpy as np


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def patch_fp8(model_name):
    """Em Hopper/Ada (cc>=8.9) mantém FP8 LIGADO; em GPU < 8.9 desliga interceptando a
    leitura da config do Evo2. Idempotente.

    O L4 da g6 é Ada, cc 8.9 -> FP8 permanece ligado, igual à corrida original no H100.
    Isso importa: o controle de precisão do artigo (Tabela S1) mostra que forçar bf16
    degrada bastante, então rodar em bf16 produziria embeddings não comparáveis.
    """
    import yaml, torch
    import evo2.models as M
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    want_fp8 = cc >= (8, 9)
    log(f"GPU cc {cc[0]}.{cc[1]} -> use_fp8_input_projections = {want_fp8}")
    res = M.CONFIG_MAP[model_name]
    cfg = yaml.safe_load(M.pkgutil.get_data(M.__name__, res))
    cfg["use_fp8_input_projections"] = bool(want_fp8)
    patched = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    if not hasattr(M, "_orig_get_data"):
        M._orig_get_data = M.pkgutil.get_data
    _orig = M._orig_get_data

    def _gd(pkg, r):
        return patched if (pkg == M.__name__ and r == res) else _orig(pkg, r)
    M.pkgutil.get_data = _gd
    return want_fp8


def resolve_weights(weights_local, model_name):
    """`Evo2(local_path=...)` espera o **arquivo .pt**, não um diretório.

    Aceita as duas formas para não repetir o erro:
      - caminho direto do .pt                     -> usa como está
      - raiz com <root>/<model>/<model>.pt        -> resolve
      - raiz com <root>/<model>.pt                -> resolve
      - None                                      -> deixa o Evo2 baixar do HF
    """
    import glob, os
    if not weights_local:
        return None
    if os.path.isfile(weights_local):
        return weights_local
    if not os.path.isdir(weights_local):
        raise FileNotFoundError(f"--weights-local não existe: {weights_local}")
    for cand in (os.path.join(weights_local, model_name, f"{model_name}.pt"),
                 os.path.join(weights_local, f"{model_name}.pt")):
        if os.path.isfile(cand):
            log(f"pesos resolvidos: {cand}")
            return cand
    hits = sorted(glob.glob(os.path.join(weights_local, "**", "*.pt"), recursive=True))
    hits = [h for h in hits if model_name in os.path.basename(h)] or hits
    if len(hits) == 1:
        log(f"pesos resolvidos: {hits[0]}")
        return hits[0]
    raise FileNotFoundError(
        f"--weights-local é um diretório e não achei o .pt de {model_name} nele. "
        f"Candidatos: {hits[:5] or 'nenhum .pt encontrado'}. "
        f"Passe o caminho do arquivo, ex.: {weights_local}/{model_name}/{model_name}.pt")


def describe_devices(inner, max_children=40):
    """Onde vivem os parâmetros de cada filho de primeiro nível. Diagnóstico para erro de
    device misto — diz QUAL módulo ficou fora do lugar."""
    import torch
    out = {}
    for name, mod in list(inner.named_children())[:max_children]:
        devs = {str(p.device) for p in mod.parameters(recurse=True)}
        devs |= {str(b.device) for b in mod.buffers(recurse=True)}
        out[name] = sorted(devs) if devs else ["<sem tensores>"]
    solto = {str(p.device) for n, p in inner.named_parameters(recurse=False)}
    if solto:
        out["<params soltos no topo>"] = sorted(solto)
    return out


def shard_pipeline(inner, devices=None, verbose=True):
    """Fatiamento manual em pipeline pelos blocos do StripedHyena.

    Por que não `accelerate.dispatch_model`: ele instala hooks que movem os *inputs* de
    cada submódulo para o device daquele submódulo, o que pressupõe que todo o fluxo de
    dados passa pelas assinaturas de forward. O StripedHyena não satisfaz isso e o forward
    morre com "Expected all tensors to be on the same device" mesmo carregando bem.

    Aqui a divisão é explícita e mínima:
      - acha a ModuleList mais longa (os blocos), sem depender do nome;
      - distribui os blocos entre as GPUs em fatias contíguas (pipeline);
      - todo o resto (embedding, norm final, unembed) fica no device 0;
      - um pre-hook por bloco move o estado oculto para o device do bloco, e um hook no
        último bloco traz o resultado de volta para o device 0.

    Pressupõe que o único dado que atravessa blocos é o tensor de estado oculto. Se houver
    tensor guardado como atributo solto (não parâmetro nem buffer), ele não se move com
    `.to()` e o forward vai falhar — `describe_devices` ajuda a achar.
    """
    import torch, torch.nn as nn
    if devices is None:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) == 1:
        inner.to(devices[0]); return {"mode": "single", "devices": devices}

    name_blocks, blocks = None, None
    for name, mod in inner.named_children():
        if isinstance(mod, nn.ModuleList) and (blocks is None or len(mod) > len(blocks)):
            name_blocks, blocks = name, mod
    if blocks is None or len(blocks) < len(devices):
        raise RuntimeError("não achei uma ModuleList de blocos grande o bastante para fatiar")

    n, nd = len(blocks), len(devices)

    # A GPU 0 carrega embedding + norm final + unembed ALÉM da sua fatia de blocos, então
    # divisão igual a sobrecarrega: no 20B em 4x L40S, com 6 blocos por card, o OOM em 32k
    # acontece na GPU 0 (alocação única de 16 GB) enquanto as outras têm folga. head_relief
    # tira N blocos da GPU 0 e redistribui. Ajustável por EVO2_HEAD_RELIEF.
    # DEFAULT 0 = divisão igual, que é a configuração MEDIDA como boa (20B ate 16k na
    # g6e.12xlarge). head_relief=2 deu "illegal memory access" ja em 4k: os kernels do
    # StripedHyena alocam workspace na GPU CORRENTE, nao na GPU do tensor (o proprio vortex
    # loga "Allocating cublas workspace for device=N"), entao mexer no balanceamento sem
    # acertar o device corrente quebra. Ver EVO2_SET_DEVICE abaixo.
    head_relief = int(os.environ.get("EVO2_HEAD_RELIEF", "0"))
    base = n // nd
    quotas = [base] * nd
    for i in range(n - base * nd):
        quotas[nd - 1 - i] += 1
    if head_relief > 0 and nd > 1:
        move = max(0, min(head_relief, quotas[0] - 1))
        quotas[0] -= move
        for i in range(move):
            quotas[1 + (i % (nd - 1))] += 1
    assign = []
    for d, q in zip(devices, quotas):
        assign += [d] * q
    assert len(assign) == n

    # resto no device 0
    for name, mod in inner.named_children():
        if name != name_blocks:
            mod.to(devices[0])
    for _, p in inner.named_parameters(recurse=False):
        p.data = p.data.to(devices[0])

    # EVO2_SET_DEVICE=1 faz o pre-hook trocar tambem a GPU CORRENTE, nao so mover o tensor.
    # Necessario se os kernels alocarem workspace no current_device (hipotese para o
    # "illegal memory access" com balanceamento desigual). OFF por padrao: a configuracao
    # equilibrada ja funciona sem isso, e ligar mexe no caminho que esta validado.
    set_dev = os.environ.get("EVO2_SET_DEVICE", "0") == "1"

    def mover(dev):
        idx = int(str(dev).split(":")[1]) if ":" in str(dev) else 0

        def pre(_m, args, kwargs):
            if set_dev:
                torch.cuda.set_device(idx)
            args = tuple(a.to(dev) if torch.is_tensor(a) else a for a in args)
            kwargs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            return args, kwargs
        return pre

    for i, blk in enumerate(blocks):
        blk.to(assign[i])
        blk.register_forward_pre_hook(mover(assign[i]), with_kwargs=True)

    def back(_m, _i, output):
        if set_dev:
            torch.cuda.set_device(0)
        if torch.is_tensor(output):
            return output.to(devices[0])
        if isinstance(output, tuple):
            return tuple(o.to(devices[0]) if torch.is_tensor(o) else o for o in output)
        return output
    blocks[-1].register_forward_hook(back)

    layout = {d: assign.count(d) for d in devices}
    if verbose:
        log(f"pipeline manual: {n} blocos ({name_blocks}) em {nd} GPUs -> {layout}")
    return {"mode": "pipeline", "devices": devices, "n_blocks": n,
            "blocks_attr": name_blocks, "layout": layout}


def make_windows(seq, window, stride, max_windows):
    L = len(seq)
    if L <= window:
        return [seq]
    starts = list(range(0, L - window + 1, stride))
    if starts[-1] != L - window:
        starts.append(L - window)
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = sorted(set(starts[j] for j in idx))
    return [seq[s:s + window] for s in starts]


def read_fasta(path, wanted=None):
    """FASTA -> dict. `wanted` (set) filtra por accession."""
    out, acc, buf = {}, None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if acc is not None and (wanted is None or acc in wanted):
                    out[acc] = "".join(buf)
                acc = ln[1:].split()[0]; buf = []
            else:
                buf.append(ln.strip())
    if acc is not None and (wanted is None or acc in wanted):
        out[acc] = "".join(buf)
    return out


def mean_pool_embed(model, layer, seq, window, stride, max_windows, device):
    """Mean-pool do embedding de UMA camada, sobre as janelas — mesma mecânica de
    `multi_layer_embed` em sweep_layers_20b.py, com hook que já reduz na GPU."""
    import torch
    seq = seq.upper()
    pooled, acc = {}, []

    def hook(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        pooled["v"] = out.detach().float().mean(dim=1).squeeze(0).cpu().numpy()

    h = model.model.get_submodule(layer).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for w in make_windows(seq, window, stride, max_windows):
                ids = torch.tensor(model.tokenizer.tokenize(w),
                                   dtype=torch.int).unsqueeze(0).to(device)
                model.model.forward(ids)
                acc.append(pooled["v"])
                del ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        h.remove()
    return np.mean(acc, axis=0)
