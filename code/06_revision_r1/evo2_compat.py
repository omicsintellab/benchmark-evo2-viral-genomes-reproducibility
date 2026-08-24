#!/usr/bin/env python3
"""evo2_compat.py — helpers shared by the revision R1 scripts.

Faithful copy of `patch_fp8` and `make_windows` from
`code/02_embeddings/probe_evo2_viral.py`, so that the revision scripts run standalone on a GPU
host without cloning the whole repository. Keep in sync with the original: the window, stride
and max_windows settings define comparability with the cached embeddings.
"""
import os
import time
import numpy as np


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def patch_fp8(model_name):
    """On Hopper/Ada (cc >= 8.9) keeps FP8 ON; on cards below 8.9 disables it by intercepting the
    leitura da config do Evo2. Idempotente.

    This matters: the precision control of the paper (Supplementary Table S3) shows that
    forcing bf16 degrades performance substantially, so a bf16 run would produce embeddings
    that are not comparable with the cached ones.
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
    """`Evo2(local_path=...)` expects the **.pt file**, not a directory.

    Both forms are accepted so the mistake is not repeated:
      - direct path to the .pt                    -> used as is
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
        raise FileNotFoundError(f"--weights-local does not exist: {weights_local}")
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
        f"--weights-local is a directory and the .pt of {model_name} was not found in it. "
        f"Candidatos: {hits[:5] or 'nenhum .pt encontrado'}. "
        f"Passe o caminho do arquivo, ex.: {weights_local}/{model_name}/{model_name}.pt")


def describe_devices(inner, max_children=40):
    """Where the parameters of each top-level child live. Diagnostic for mixed-device errors:
    it says WHICH module ended up in the wrong place."""
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

    Why not `accelerate.dispatch_model`: it installs hooks that move each submodule's *inputs*
    to that submodule's device, which assumes all data flow passes through forward
    signatures. StripedHyena does not satisfy that, and the forward
    morre com "Expected all tensors to be on the same device" mesmo carregando bem.

    Here the split is explicit and minimal:
      - acha a ModuleList mais longa (os blocos), sem depender do nome;
      - blocks are distributed across GPUs in contiguous slices (pipeline);
      - todo o resto (embedding, norm final, unembed) fica no device 0;
      - um pre-hook por bloco move o estado oculto para o device do bloco, e um hook no
        the last block brings the result back to device 0.

    It assumes the only data crossing blocks is the hidden-state tensor. A tensor held as a
    loose attribute (neither parameter nor buffer) will not move with
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
        raise RuntimeError("no ModuleList of blocks large enough to shard was found")

    n, nd = len(blocks), len(devices)

    # GPU 0 carries the embedding, the final norm and the unembed BESIDES its slice of blocks,
    # so an equal split overloads it: the OOM at 32k happens on GPU 0 (a single 16 GB
    # allocation) while the others have room. head_relief takes N blocks off GPU 0 and
    # redistributes them; adjustable through EVO2_HEAD_RELIEF.
    # DEFAULT 0 = equal split, which is the configuration MEASURED as good (20B up to 16k in
    # a 4-card setup). head_relief=2 gave an "illegal memory access" already at 4k: the
    # StripedHyena kernels allocate workspace on the CURRENT GPU, not on the tensor's GPU
    # (vortex itself logs "Allocating cublas workspace for device=N"), so changing the balance
    # without also setting the current device breaks. See EVO2_SET_DEVICE below.
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

    # everything else on device 0
    for name, mod in inner.named_children():
        if name != name_blocks:
            mod.to(devices[0])
    for _, p in inner.named_parameters(recurse=False):
        p.data = p.data.to(devices[0])

    # EVO2_SET_DEVICE=1 makes the pre-hook switch the CURRENT GPU as well, not just move the
    # tensor. Needed if the kernels allocate workspace on current_device (the hypothesis for
    # the "illegal memory access" under an uneven balance). OFF by default: the even split
    # works without it, and turning it on changes the validated path.
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
    """Mean-pool the embedding of ONE layer over the windows: same mechanics as
    `multi_layer_embed` in sweep_layers_20b.py, with a hook that reduces on the GPU."""
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
