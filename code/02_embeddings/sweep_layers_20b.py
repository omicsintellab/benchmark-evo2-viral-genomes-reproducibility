#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_layers_20b.py — Varredura de camadas do EVO2 20B para probing.

Motivação: no probing inicial (2026-07-07) o 20B na camada blocks.21.mlp.l3 NÃO
superou o 7B em nenhum alvo. Hipótese principal: **camada de embedding subótima**
(blocks.21 foi escolhida por profundidade relativa 21/24≈28/32 do 7B, mas a camada
ótima é empírica por modelo). Este script testa a hipótese:

  - extrai embeddings de VÁRIAS camadas NUM ÚNICO forward (hooks que fazem mean-pool
    na hora -> não retém as ativações completas -> memória ~igual à de 1 camada);
  - cacheia por camada (embeddings_evo2_20b_L{n}_w{W}.npz);
  - roda a MESMA suíte de probes por camada -> ranking de camadas;
  - compara cada camada ao baseline 7B (mesmas accessions, split idêntico).

Probes reimplementados aqui são IDÊNTICOS aos de probe_evo2_viral.py (mesmos
estimadores, CV, preprocessing) + n_jobs=-1 (só paraleliza as folds; não muda
resultado). A camada blocks.21 no output serve de reprodução/validação: deve bater
~0.835 de Baltimore acc do run anterior.

Uso (dentro do container evo2, GPU H100):
  python -u sweep_layers_20b.py --weights-local /path/to/weights/evo2_20b/evo2_20b.pt \
    --local-dir /path/to/workdir --out-s3 experiments/embed_probes/
Dry-run:
  python -u sweep_layers_20b.py --weights-local <.pt> --limit 24 --no-upload
"""
import argparse, gc, json, os, time
import numpy as np

# reusa helpers determinísticos do probe principal (mesmo diretório)
from probe_evo2_viral import (
    patch_fp8, make_windows, load_codon_cds,
    codon_enc_genome, gc_regional_std, recomb_ir_density,
    MANIFEST_KEY, FEATURES_KEY, FASTA_KEY, EMB7B_BALT_S3, EMB7B_FEAT_S3,
    BALT_NAMES, log,
)

def parse_args():
    p = argparse.ArgumentParser(description="Varredura de camadas do EVO2 20B (probing).")
    p.add_argument("--weights-local", required=True, help="Caminho do .pt do 20B.")
    p.add_argument("--bucket", default=os.environ.get("EVO2_BENCH_BUCKET", ""),
                   help="Bucket S3 opcional para staging de inputs/outputs. Vazio = tudo local.")
    p.add_argument("--model", default="evo2_20b")
    p.add_argument("--layers",
                   default="blocks.11.mlp.l3,blocks.15.mlp.l3,blocks.18.mlp.l3,blocks.21.mlp.l3,blocks.23.mlp.l3",
                   help="Camadas a varrer (CSV). blocks.21 reproduz o run anterior.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--window", type=int, default=32768)
    p.add_argument("--stride", type=int, default=16384)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-dir", default="/path/to/workdir")
    p.add_argument("--out-s3", default="experiments/embed_probes/")
    p.add_argument("--limit", type=int, default=0, help="Só os N primeiros genomas (dry-run).")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--fp8", default="auto", choices=["auto", "off"],
                   help="auto = FP8 ligado em Hopper/Ada (default do patch_fp8); off = força bf16 "
                        "(controle de precisão p/ comparar com o 7B, que rodou em bf16). "
                        "Com off, os caches ganham sufixo _bf16 p/ não colidir com os de FP8.")
    return p.parse_args()

def patch_fp8_force(model_name, want_fp8):
    """Igual ao patch_fp8 do probe_evo2_viral, mas com want_fp8 FORÇADO (ignora a cc da GPU).
    Usado no controle de precisão: forçar bf16 (want_fp8=False) num H100 p/ casar com o 7B."""
    import yaml
    import evo2.models as M
    log(f"patch_fp8_force: use_fp8_input_projections = {want_fp8} (forçado)")
    res = M.CONFIG_MAP[model_name]
    cfg = yaml.safe_load(M.pkgutil.get_data(M.__name__, res))
    cfg["use_fp8_input_projections"] = bool(want_fp8)
    patched = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    if not hasattr(M, "_orig_get_data"): M._orig_get_data = M.pkgutil.get_data
    _orig = M._orig_get_data
    def _gd(pkg, r): return patched if (pkg == M.__name__ and r == res) else _orig(pkg, r)
    M.pkgutil.get_data = _gd
    return want_fp8

# ----------------------------------------------------------------------------- probes (idênticos + n_jobs=-1)
def cv_r2(Xm, yv, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, KFold
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1,1,10,100,1000]))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, Xm, yv, cv=cv, scoring="r2", n_jobs=-1).mean())

def probe_baltimore(X, y, gc, length, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    yp = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, yp); f1 = f1_score(y, yp, average="macro")
    Xtriv = np.column_stack([gc, length])
    yp_t = cross_val_predict(clf, Xtriv, y, cv=cv, n_jobs=-1)
    acc_t = accuracy_score(y, yp_t)
    return dict(acc=round(acc,3), f1=round(f1,3), acc_gc_len=round(acc_t,3), gain=round(acc-acc_t,3))

def probe_features(X, feat_df, accs, seed, targets):
    lenGC_full = np.column_stack([np.log1p(feat_df["genome_length"].values), feat_df["GC"].values])
    rows = []
    for f, logt in targets:
        if f not in feat_df.columns: continue
        y = feat_df[f].values.astype(float)
        m = np.isfinite(y)
        if m.sum() < 50: continue
        yv = np.log1p(np.clip(y[m],0,None)) if logt else y[m]
        Xm = X[m]; lenGC = lenGC_full[m]
        r2_lenGC = cv_r2(lenGC, yv, seed)
        r2_emb   = cv_r2(Xm, yv, seed)
        r2_comb  = cv_r2(np.hstack([Xm, lenGC]), yv, seed)
        rows.append(dict(feature=f, n=int(m.sum()), R2_lenGC=round(r2_lenGC,3),
                         R2_emb=round(r2_emb,3), R2_comb=round(r2_comb,3),
                         R2_inc=round(r2_comb-r2_lenGC,3)))
    return rows

def probe_classification(X, labels, seed, valid):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    m = pd.Series(labels).isin(valid).values
    if m.sum() < 50: return None
    Xc, yc = X[m], np.asarray(labels)[m]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    yp = cross_val_predict(clf, Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
    return dict(n=int(m.sum()), n_classes=len(set(yc.tolist())),
                acc=round(accuracy_score(yc,yp),3), f1=round(f1_score(yc,yp,average="macro"),3))

# ----------------------------------------------------------------------------- embeddings multi-camada (1 forward)
def multi_layer_embed(model, layers, seq, window, stride, max_windows, device):
    """Extrai TODAS as camadas num único forward por janela. Hooks fazem mean-pool
    (dim seq) e movem p/ CPU na hora -> não retém as ativações completas na GPU
    (memória ~igual à de 1 camada). Depois faz mean-pool das janelas por camada."""
    import torch
    seq = seq.upper()
    accum = {L: [] for L in layers}
    pooled = {}
    def mk(name):
        def hook(_m, _i, output):
            out = output[0] if isinstance(output, tuple) else output
            pooled[name] = out.detach().float().mean(dim=1).squeeze(0).cpu().numpy()
        return hook
    handles = [model.model.get_submodule(L).register_forward_hook(mk(L)) for L in layers]
    try:
        with torch.inference_mode():
            for w in make_windows(seq, window, stride, max_windows):
                ids = torch.tensor(model.tokenizer.tokenize(w), dtype=torch.int).unsqueeze(0).to(device)
                model.model.forward(ids)
                for L in layers:
                    accum[L].append(pooled[L])
                del ids
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        for h in handles: h.remove()
    return {L: np.mean(accum[L], axis=0) for L in layers}

# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    import pandas as pd, boto3
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    os.makedirs(args.local_dir, exist_ok=True)
    os.makedirs(f"{args.local_dir}/results", exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    tag = args.model
    # sufixo de precisão nos artefatos: "" (FP8, retrocompatível) ou "_bf16" (controle)
    prec_suffix = "_bf16" if args.fp8 == "off" else ""
    log(f"varredura de camadas: {layers} | precisão: {'bf16 (forçado)' if args.fp8=='off' else 'auto/FP8'}")

    # ---- 1. dados do corpus
    for key in (MANIFEST_KEY, FEATURES_KEY, FASTA_KEY):
        dst = f"{args.local_dir}/{os.path.basename(key)}"
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            log("baixando", key); s3.download_file(args.bucket, key, dst)
    man  = pd.read_parquet(f"{args.local_dir}/{os.path.basename(MANIFEST_KEY)}")
    feat = pd.read_parquet(f"{args.local_dir}/{os.path.basename(FEATURES_KEY)}")
    feat_cols = [c for c in ["accession","GC","genome_length","coding_fraction","gene_density",
                 "n_genes","n_CDS","noncoding_bp","mean_intergenic_len"]
                 if c in feat.columns and (c=="accession" or c not in man.columns)]
    meta = man.merge(feat[feat_cols], on="accession", how="left")

    # ---- 2. reuso das accessions do 7B (comparação limpa) + embeddings 7B p/ referência
    bdst = f"{args.local_dir}/{os.path.basename(EMB7B_BALT_S3)}"
    fdst = f"{args.local_dir}/{os.path.basename(EMB7B_FEAT_S3)}"
    if not os.path.exists(bdst): s3.download_file(args.bucket, EMB7B_BALT_S3, bdst)
    if not os.path.exists(fdst): s3.download_file(args.bucket, EMB7B_FEAT_S3, fdst)
    db = np.load(bdst, allow_pickle=True); X7b_balt = db["X"].astype(np.float32)
    balt_accs = db["accs"].astype(str); balt_y = db["y"].astype(str)
    dfe = np.load(fdst, allow_pickle=True); X7b_feat = dfe["X"].astype(np.float32)
    feat_accs = dfe["accs"].astype(str)
    log(f"reuso 7B: baltimore {len(balt_accs)} accs, features {len(feat_accs)} accs")

    wanted = sorted(set(balt_accs.tolist()) | set(feat_accs.tolist()))
    if args.limit:
        wanted = wanted[:args.limit]; wset = set(wanted)
        keepb = np.array([a in wset for a in balt_accs]); balt_accs, balt_y, X7b_balt = balt_accs[keepb], balt_y[keepb], X7b_balt[keepb]
        keepf = np.array([a in wset for a in feat_accs]); feat_accs, X7b_feat = feat_accs[keepf], X7b_feat[keepf]
    log(f"genomas a embedar: {len(wanted)} (limit={args.limit or 'off'})")

    # ---- 3. FASTA do subset
    fa = f"{args.local_dir}/{os.path.basename(FASTA_KEY)}"
    wset = set(wanted); seqs = {}; acc=None; buf=[]
    def flush():
        if acc is not None and acc in wset: seqs[acc] = "".join(buf)
    with open(fa) as fh:
        for ln in fh:
            if ln.startswith(">"): flush(); acc = ln[1:].split()[0]; buf=[]
            else: buf.append(ln.strip())
    flush()
    log(f"sequências carregadas: {len(seqs)}/{len(wanted)}")

    # ---- 4. embeddings multi-camada (1 forward por janela) — por camada, cacheia
    emb = {L: {} for L in layers}   # emb[layer][acc] = vetor
    cache_paths = {L: f"{args.local_dir}/embeddings_{tag}_L{L.split('.')[1]}{prec_suffix}_w{args.window}.npz" for L in layers}
    missing = [L for L in layers if not os.path.exists(cache_paths[L])]
    if not missing:
        for L in layers:
            d = np.load(cache_paths[L], allow_pickle=True)
            emb[L] = {a: v for a, v in zip(d["accs"].astype(str), d["X"].astype(np.float32))}
        log(f"todas as {len(layers)} camadas do cache")
    else:
        log(f"carregando modelo {args.model} ...")
        (patch_fp8_force(args.model, False) if args.fp8 == "off" else patch_fp8(args.model)); gc.collect()
        from evo2 import Evo2
        import torch
        model = Evo2(args.model, local_path=args.weights_local)
        model.model.eval()
        if torch.cuda.is_available():
            log(f"GPU {torch.cuda.get_device_name(0)} | VRAM pós-load {torch.cuda.memory_allocated()/1e9:.1f} GB")
        t0 = time.time()
        for i, a in enumerate(wanted):
            if a not in seqs: continue
            try:
                vecs = multi_layer_embed(model, layers, seqs[a], args.window, args.stride, args.max_windows, args.device)
                for L in layers: emb[L][a] = vecs[L]
            except Exception as e:
                log("falhou", a, "->", repr(e))
            if (i+1) % 25 == 0:
                log(f"  {i+1}/{len(wanted)} | {(time.time()-t0)/(i+1):.2f}s/genoma")
        for L in layers:
            accs_arr = np.array(list(emb[L].keys()))
            X_arr = np.vstack([emb[L][a] for a in accs_arr]).astype(np.float32)
            np.savez_compressed(cache_paths[L], X=X_arr, accs=accs_arr)
            log(f"cache {L}: {cache_paths[L]} {X_arr.shape}")
            if not args.no_upload:
                key = f"{args.out_s3.rstrip('/')}/embeddings/{os.path.basename(cache_paths[L])}"
                try: s3.upload_file(cache_paths[L], args.bucket, key); log("  -> S3", key)
                except Exception as e: log("  aviso: upload falhou:", e)
        del model; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

    # ---- 5. features de sequência (model-independent) + codon_enc_cds
    def clean(a): return "".join(c for c in seqs[a].upper() if c in "ACGT")
    enc_cds = load_codon_cds(s3, args.bucket, args.local_dir,
                             f"{args.local_dir}/codon_enc_cds.parquet",
                             np.array(wanted), False, args.no_upload)
    targets = [("coding_fraction",False),("gene_density",False),("n_genes",True),
               ("noncoding_bp",True),("mean_intergenic_len",True),
               ("codon_enc_cds",False),("codon_enc_genome",False),
               ("gc_regional_std",False),("recomb_ir_density",True)]

    # fdf base (features alinhadas às feat_accs) — vale p/ todas as camadas
    def build_fdf(af):
        fdf = meta.set_index("accession").reindex(af).copy()
        fdf["codon_enc_genome"]  = [codon_enc_genome(clean(a)) for a in af]
        fdf["gc_regional_std"]   = [gc_regional_std(clean(a)) for a in af]
        fdf["recomb_ir_density"] = [recomb_ir_density(clean(a)) for a in af]
        if enc_cds is not None:
            fdf["codon_enc_cds"] = enc_cds["codon_enc_cds"].reindex(af).values
        return fdf

    # ---- 6. referência 7B (uma vez, mesmas accs/split)
    def run_suite(Xb, ab, yb, Xf, af):
        fb = meta.set_index("accession").reindex(ab)
        res_b = probe_baltimore(Xb, yb, fb["GC"].fillna(0).values, fb["genome_length"].fillna(0).values, args.seed)
        fdf = build_fdf(af)
        rows = probe_features(Xf, fdf, af, args.seed, targets)
        host = probe_classification(Xf, fdf["host"].values, args.seed, {"eukaryote","bacteria","archaea"})
        fams = fdf["family"].replace("", np.nan).dropna().value_counts()
        top = fams[fams>=25].index.tolist()
        fam = probe_classification(Xf, fdf["family"].values, args.seed, set(top)) if len(top)>=2 else None
        return {"baltimore": res_b, "features": {r["feature"]: r["R2_inc"] for r in rows},
                "host": host, "family": fam, "n_baltimore": len(ab), "n_features": len(af)}

    # alinhar 7B: usar exatamente as accs presentes nos caches 7B (já são balt_accs/feat_accs)
    log("probes 7B (referência) ...")
    ref7b = run_suite(X7b_balt, balt_accs, balt_y, X7b_feat, feat_accs)
    log("7B:", {"baltimore": ref7b["baltimore"]["acc"], "host": ref7b["host"], "features": ref7b["features"]})

    # ---- 7. probes por camada do 20B
    def stack(emb_L, accs_list):
        ok = [a for a in accs_list if a in emb_L]
        return np.vstack([emb_L[a] for a in ok]).astype(np.float32), np.array(ok)

    per_layer = {}
    for L in layers:
        Xb, ab = stack(emb[L], balt_accs.tolist())
        yb = pd.Series(balt_y, index=balt_accs).reindex(ab).to_numpy().astype(str)
        Xf, af = stack(emb[L], feat_accs.tolist())
        res = run_suite(Xb, ab, yb, Xf, af)
        per_layer[L] = res
        log(f"[{L}] Baltimore {res['baltimore']['acc']} (gain {res['baltimore']['gain']:+}) | "
            f"host {res['host']['acc'] if res['host'] else None} | "
            f"codon_enc_cds {res['features'].get('codon_enc_cds')}")

    # ---- 8. salvar + resumo
    out = {"model": tag, "window": args.window, "layers": layers,
           "precision": ("bf16" if args.fp8 == "off" else "fp8"),
           "ref_7b": ref7b, "per_layer": per_layer}
    out_json = f"{args.local_dir}/results/sweep_layers_{tag}{prec_suffix}.json"
    with open(out_json, "w") as f: json.dump(out, f, indent=2, default=str)
    if not args.no_upload:
        try:
            s3.upload_file(out_json, args.bucket, f"{args.out_s3.rstrip('/')}/results/sweep_layers_{tag}{prec_suffix}.json")
            log("resultados -> S3")
        except Exception as e: log("aviso: upload resultados falhou:", e)

    def cell(v, w):
        if v is None: return f"{'-':>{w}}"
        if isinstance(v, float): return f"{v:>{w}.3f}"
        return f"{v:>{w}}"
    def g(res, k): return res["features"].get(k)
    def row(name, r):
        host = r["host"]["acc"] if r["host"] else None
        return (f"{name:<20}" + cell(r['baltimore']['acc'],10) + cell(r['baltimore']['gain'],11)
                + cell(host,8) + cell(g(r,'codon_enc_cds'),11)
                + cell(g(r,'coding_fraction'),11) + cell(g(r,'gene_density'),10))
    print("\n" + "="*78 + f"\nVARREDURA DE CAMADAS — {tag}  (7B ref: Baltimore "
          f"{ref7b['baltimore']['acc']}, host {ref7b['host']['acc'] if ref7b['host'] else '-'})\n" + "="*78)
    print(f"{'layer':<20}{'Balt_acc':>10}{'Balt_gain':>11}{'host':>8}{'codon_cds':>11}{'coding_fr':>11}{'gene_den':>10}")
    for L in layers:
        print(row(L, per_layer[L]))
    print("-"*78)
    print(row("7B (ref)", ref7b))
    print("="*78)
    # melhor camada por Baltimore
    best = max(layers, key=lambda L: per_layer[L]["baltimore"]["acc"])
    print(f"Melhor camada (Baltimore acc): {best} = {per_layer[best]['baltimore']['acc']} "
          f"(vs 7B {ref7b['baltimore']['acc']}); blocks.21 = {per_layer.get('blocks.21.mlp.l3',{}).get('baltimore',{}).get('acc')} (reprodução)")

if __name__ == "__main__":
    main()
