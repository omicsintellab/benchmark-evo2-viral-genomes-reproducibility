#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation_completion_20b.py — Porta headless do eixo 3 (geração/completação) do
benchmark, do evo2_7b (notebook viral_generation_completion_evo2_7b_executado.ipynb)
para o evo2_20b. Metodologia IDÊNTICA (mesmos hiperparâmetros, mesma seleção de
accessions via seed=42 sobre o mesmo manifest -> mesmos 200 genomas do 7B), para
que a comparação 7B x 20B seja limpa.

Dois experimentos:
  1) Perplexidade (bits/nt teacher-forced) em euk held-out (fora do treino do
     Evo2 por design) vs fago visto (comparador).
  2) Completação de fragmentos: prompt=4096bp -> gera 1024bp; compara com o gap
     real via bits/nt teacher-forced e similaridade de k-mers (4-mer cosine),
     contra baseline Markov-4 ajustado nos prompts.

Diferença vs o notebook 7B: aqui o FP8 fica LIGADO (patch_fp8 automático por
capability da GPU — H100 = Hopper = FP8 nativo), consistente com os demais
scripts do 20B rodados hoje (probe_evo2_viral.py, sweep_layers_20b.py). O 7B
rodou em bf16 forçado (GPU L40S, não-Hopper).

Uso (dentro do container evo2, H100):
  python -u generation_completion_20b.py --weights-local /path/to/weights/evo2_20b/evo2_20b.pt \
    --local-dir /path/to/workdir --out-s3 experiments/embed_probes/
Dry-run:
  python -u generation_completion_20b.py --weights-local <.pt> --limit 8 --no-upload
"""
import argparse, itertools, os, random, time, json
from collections import defaultdict, Counter
import numpy as np, pandas as pd

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="evo2_20b")
    p.add_argument("--weights-local", required=True)
    p.add_argument("--bucket", default=os.environ.get("EVO2_BENCH_BUCKET", ""),
                   help="Bucket S3 opcional para staging de inputs/outputs. Vazio = tudo local.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--local-dir", default="/path/to/workdir")
    p.add_argument("--out-s3", default="experiments/embed_probes/")
    p.add_argument("--n-heldout", type=int, default=100, help="vírus de eucariotos (held-out)")
    p.add_argument("--n-seen", type=int, default=100, help="fagos (vistos no treino); 0 desliga")
    p.add_argument("--perp-window", type=int, default=8192)
    p.add_argument("--prompt-bp", type=int, default=4096)
    p.add_argument("--gap-bp", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--markov-order", type=int, default=4)
    p.add_argument("--kmer-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="dry-run: só os N primeiros do subset")
    p.add_argument("--no-upload", action="store_true")
    return p.parse_args()

MANIFEST_KEY = "corpus/raw/v1_refseq/composed/composition_manifest.parquet"
FASTA_KEY    = "corpus/raw/v1_refseq/fasta/all_genomes.fasta"

def patch_fp8(model_name):
    import yaml, torch
    import evo2.models as M
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    want_fp8 = cc >= (8, 9)
    log(f"GPU cc {cc[0]}.{cc[1]} -> use_fp8_input_projections = {want_fp8}")
    res = M.CONFIG_MAP[model_name]
    cfg = yaml.safe_load(M.pkgutil.get_data(M.__name__, res))
    cfg["use_fp8_input_projections"] = bool(want_fp8)
    patched = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    if not hasattr(M, "_orig_get_data"): M._orig_get_data = M.pkgutil.get_data
    _orig = M._orig_get_data
    def _gd(pkg, r): return patched if (pkg == M.__name__ and r == res) else _orig(pkg, r)
    M.pkgutil.get_data = _gd
    return want_fp8

def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.local_dir, exist_ok=True); os.makedirs(f"{a.local_dir}/results", exist_ok=True)
    import boto3, torch
    s3 = boto3.client("s3", region_name="us-east-1")

    # ---- 1. corpus + mesma seleção (seed=42) do notebook 7B ----
    man_dst = f"{a.local_dir}/{os.path.basename(MANIFEST_KEY)}"
    if not os.path.exists(man_dst): s3.download_file(a.bucket, MANIFEST_KEY, man_dst)
    fa = f"{a.local_dir}/{os.path.basename(FASTA_KEY)}"
    man = pd.read_parquet(man_dst)
    need = max(a.prompt_bp + a.gap_bp, a.perp_window)
    man = man[man["length"] >= need]
    euk = man[man["host"] == "eukaryote"].sample(min(a.n_heldout, (man.host == "eukaryote").sum()), random_state=a.seed)
    sel = euk.assign(set_label="euk_heldout")
    if a.n_seen > 0:
        phg = man[man["quota_group"] == "phage"].sample(min(a.n_seen, (man.quota_group == "phage").sum()), random_state=a.seed)
        sel = pd.concat([sel, phg.assign(set_label="phage_seen")])
    subset = sel.reset_index(drop=True)
    log(f"selecionados: {len(subset)} (need>={need} bp)")
    log(subset.groupby("set_label").size().to_string())

    wanted = set(subset["accession"]); seqs = {}; acc = None; buf = []
    def flush():
        if acc is not None and acc in wanted: seqs[acc] = "".join(buf)
    with open(fa) as fh:
        for ln in fh:
            if ln.startswith(">"): flush(); acc = ln[1:].split()[0]; buf = []
            else: buf.append(ln.strip())
    flush()
    subset = subset[subset["accession"].isin(seqs)].reset_index(drop=True)
    if a.limit: subset = subset.iloc[:a.limit].reset_index(drop=True)
    log(f"sequências carregadas: {len(seqs)} | subset final: {len(subset)}")

    # ---- 2. modelo ----
    log(f"carregando {a.model} ...")
    patch_fp8(a.model)
    from evo2 import Evo2
    model = Evo2(a.model, local_path=a.weights_local)
    model.model.eval()
    if torch.cuda.is_available():
        log(f"GPU {torch.cuda.get_device_name(0)} | VRAM pós-load {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def _forward_logits(ids):
        out = model(ids)
        while not isinstance(out, torch.Tensor):
            out = out[0] if isinstance(out, (tuple, list)) else next(iter(out.values()))
        return out

    @torch.inference_mode()
    def bits_per_nt(seq, region=None):
        seq = seq.upper()
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(a.device)
        logits = _forward_logits(ids)
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:].long()
        blp = -(logp.gather(1, tgt[:, None]).squeeze(1)) / np.log(2)
        if region:
            s, e = region; blp = blp[max(0, s-1):e-1]
        val = blp.mean().item()
        del ids, logits, logp
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return val

    # ---- 3. perplexidade ----
    log("=== perplexidade ===")
    rows = []
    t0 = time.time()
    for i, r in enumerate(subset.itertuples(index=False)):
        s = seqs[r.accession][:a.perp_window]
        try:
            rows.append(dict(accession=r.accession, set_label=r.set_label, baltimore=r.baltimore,
                             host=r.host, bits_nt=bits_per_nt(s)))
        except Exception as e:
            log("falhou (perplexidade)", r.accession, repr(e))
        if (i+1) % 20 == 0:
            log(f"  {i+1}/{len(subset)} | {(time.time()-t0)/(i+1):.2f}s/genoma")
    perp = pd.DataFrame(rows)
    log(perp.groupby("set_label")["bits_nt"].describe()[["count", "mean", "std"]].round(3).to_string())

    # ---- 4. completação ----
    log("=== completação (Markov baseline + geração EVO2) ===")
    K = a.kmer_k
    def fit_markov(seqlist, k):
        trans = defaultdict(Counter)
        for s in seqlist:
            s = "".join(c for c in s.upper() if c in "ACGT")
            for i in range(len(s)-k): trans[s[i:i+k]][s[i+k]] += 1
        P = {}
        for ctx, cnt in trans.items():
            tot = sum(cnt.values())+4; P[ctx] = {b: (cnt.get(b, 0)+1)/tot for b in "ACGT"}
        return P
    def markov_gen(prefix, n, k, P):
        s = prefix[-k:]; out = []
        for _ in range(n):
            pr = P.get(s[-k:])
            nxt = random.choice("ACGT") if pr is None else random.choices("ACGT", weights=[pr[b] for b in "ACGT"])[0]
            out.append(nxt); s += nxt
        return "".join(out)
    def markov_bits(prefix, gap, k, P):
        full = prefix[-k:]+gap; tot = 0
        for i in range(k, len(full)):
            p = P.get(full[i-k:i], {}).get(full[i], 0.25); tot += -np.log2(max(p, 1e-9))
        return tot/len(gap)
    KMERS = ["".join(p) for p in itertools.product("ACGT", repeat=K)]; KIDX = {k: i for i, k in enumerate(KMERS)}
    def kmer_vec(s):
        v = np.zeros(len(KMERS)); s = s.upper()
        for i in range(len(s)-K+1):
            j = KIDX.get(s[i:i+K])
            if j is not None: v[j] += 1
        n = np.linalg.norm(v); return v/n if n else v
    def kmer_cos(x, y): return float(kmer_vec(x) @ kmer_vec(y))
    def evo2_gen(prefix, n):
        out = model.generate(prompt_seqs=[prefix], n_tokens=n, temperature=a.temperature, top_k=a.top_k)
        g = getattr(out, "sequences", None)
        if g is None: g = out["sequences"] if isinstance(out, dict) else out
        g = g[0]
        if len(g) >= len(prefix) and g[:32] == prefix[:32]: g = g[len(prefix):]
        return g[:n]
    def gcpct(s): s = s.upper(); return 100*(s.count("G")+s.count("C"))/max(1, len(s))

    Pmk = fit_markov([seqs[acc][:a.prompt_bp] for acc in subset.accession], a.markov_order)

    rows = []; t0 = time.time()
    for i, r in enumerate(subset.itertuples(index=False)):
        full = seqs[r.accession]
        prefix = full[:a.prompt_bp]; true_gap = full[a.prompt_bp:a.prompt_bp+a.gap_bp]
        if len(true_gap) < a.gap_bp: continue
        try:
            b_evo = bits_per_nt(prefix+true_gap, region=(a.prompt_bp, a.prompt_bp+a.gap_bp))
            b_mk = markov_bits(prefix, true_gap, a.markov_order, Pmk)
            g_evo = evo2_gen(prefix, a.gap_bp)
            g_mk = markov_gen(prefix, a.gap_bp, a.markov_order, Pmk)
            rows.append(dict(accession=r.accession, set_label=r.set_label, baltimore=r.baltimore,
                             bits_evo=b_evo, bits_markov=b_mk,
                             kmer_evo=kmer_cos(g_evo, true_gap), kmer_markov=kmer_cos(g_mk, true_gap),
                             gc_true=gcpct(true_gap), gc_evo=gcpct(g_evo)))
        except Exception as e:
            log("falhou (completação)", r.accession, repr(e))
        if (i+1) % 10 == 0:
            log(f"  {i+1}/{len(subset)} | {(time.time()-t0)/(i+1):.2f}s/genoma")
    comp = pd.DataFrame(rows)
    log("bits/nt no trecho-alvo (menor=melhor):")
    log(comp.groupby("set_label")[["bits_evo", "bits_markov"]].mean().round(3).to_string())
    log("similaridade de k-mers gerado vs real (maior=melhor):")
    log(comp.groupby("set_label")[["kmer_evo", "kmer_markov"]].mean().round(3).to_string())

    # ---- 5. salvar ----
    perp.to_csv(f"{a.local_dir}/results/generation_perplexity_{a.model}.csv", index=False)
    comp.to_csv(f"{a.local_dir}/results/generation_completion_{a.model}.csv", index=False)
    summary = {
        "model": a.model,
        "n_subset": len(subset),
        "perplexity": {k: {"mean": float(v["mean"]), "std": float(v["std"]), "n": int(v["count"])}
                       for k, v in perp.groupby("set_label")["bits_nt"].describe()[["count", "mean", "std"]].to_dict("index").items()},
        "completion": {
            "bits_evo": comp.groupby("set_label")["bits_evo"].mean().round(3).to_dict(),
            "bits_markov": comp.groupby("set_label")["bits_markov"].mean().round(3).to_dict(),
            "kmer_evo": comp.groupby("set_label")["kmer_evo"].mean().round(3).to_dict(),
            "kmer_markov": comp.groupby("set_label")["kmer_markov"].mean().round(3).to_dict(),
        },
    }
    out_json = f"{a.local_dir}/results/generation_summary_{a.model}.json"
    json.dump(summary, open(out_json, "w"), indent=2, default=str)
    log("salvo:", out_json)
    if not a.no_upload:
        for f in [f"generation_perplexity_{a.model}.csv", f"generation_completion_{a.model}.csv", f"generation_summary_{a.model}.json"]:
            try:
                s3.upload_file(f"{a.local_dir}/results/{f}", a.bucket, f"{a.out_s3.rstrip('/')}/results/{f}")
                log("  -> S3", f)
            except Exception as e:
                log("  aviso: upload falhou:", f, e)

    print("\n" + "="*64 + f"\nRESUMO — geração/completação {a.model}\n" + "="*64)
    print(json.dumps(summary, indent=2, default=str))
    print("="*64)

if __name__ == "__main__":
    main()
