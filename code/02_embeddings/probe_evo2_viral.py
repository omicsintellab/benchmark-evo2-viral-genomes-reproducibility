#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_evo2_viral.py — headless probing benchmark of Evo 2 on viral genomes (GPU stage).

Requires a GPU with the Evo 2 runtime installed (results in the paper: NVIDIA H100 80GB for
the 20B model, FP8 input projections; NVIDIA L40S 48GB for the 7B model, FP8 disabled → bf16).

In a single pass, computing the embeddings ONCE, it:
  1) extracts windowed embeddings (32 kb windows, 16 kb stride) per genome, mean-pooling over
     tokens and then over windows -> one vector per genome;
  2) probes Baltimore class (multinomial logistic regression, 5-fold CV) + GC+length control;
  3) probes genomic features (ridge, 5-fold CV, incremental R² over length+GC):
     coding_fraction, gene_density, n_genes, noncoding_bp, mean_intergenic_len;
  4) computes sequence-derived features: codon_enc_genome (proxy), gc_regional_std,
     recomb_ir_density;
  5) computes codon_enc_cds — Wright's ENC over the real CDSs from the GenBank flat file
     (model-independent: cached to parquet and reused across model sizes);
  6) classifies host domain (euk/bact/arch) and viral family (top-N families);
  7) re-runs the SAME probes over cached 7B embeddings (same accessions) for the 20B-vs-7B
     scale comparison.

Inputs (see data/README.md): corpus manifest, genome features, genome FASTA. These are staged
locally; --bucket/--out-s3 are optional and only used if you mirror artefacts to S3.

Typical run (inside the Evo 2 container, on the GPU host):
  python probe_evo2_viral.py \
    --model evo2_20b \
    --weights-local /path/to/weights/evo2_20b/evo2_20b.pt \
    --layer blocks.18.mlp.l3

Quick dry run (few genomes, validates the end-to-end path):
  python probe_evo2_viral.py --model evo2_20b --weights-local <.pt> --limit 24 --no-upload
"""
import argparse, gc, gzip, json, os, sys, subprocess, time
from collections import Counter

# ----------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description="Probing EVO2 em genomas virais (headless).")
    p.add_argument("--model", default="evo2_20b", help="evo2_20b | evo2_7b | nome de checkpoint FT.")
    p.add_argument("--weights-local", default=None,
                   help="Caminho local do .pt (local_path do Evo2). Se omitido, Evo2 baixa do HF.")
    p.add_argument("--bucket", default=os.environ.get("EVO2_BENCH_BUCKET", ""),
                   help="Bucket S3 opcional para staging de inputs/outputs. Vazio = tudo local.")
    p.add_argument("--layer", default="blocks.28.mlp.l3", help="Camada de embedding.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--window", type=int, default=32768)
    p.add_argument("--stride", type=int, default=16384)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-dir", default=os.path.expanduser("~/evo2_probe"))
    p.add_argument("--out-s3", default="experiments/embed_probes/",
                   help="Prefixo S3 (no bucket) p/ embeddings + resultados.")
    p.add_argument("--reuse-7b-accs", action="store_true", default=True,
                   help="Reusa as MESMAS accessions do cache 7B (comparação limpa). Default: ligado.")
    p.add_argument("--fresh-sample", dest="reuse_7b_accs", action="store_false",
                   help="Amostra subset do zero (quota_group/baltimore) em vez de reusar accs do 7B.")
    p.add_argument("--n-per-group", type=int, default=120, help="Usado só no modo --fresh-sample.")
    p.add_argument("--skip-codon-cds", action="store_true",
                   help="Pula o codon_enc_cds (evita baixar o gbff ~450MB se ainda não houver cache).")
    p.add_argument("--limit", type=int, default=0, help="Processa só os N primeiros genomas (dry-run).")
    p.add_argument("--no-upload", action="store_true", help="Não sobe nada para o S3.")
    p.add_argument("--no-compare-7b", action="store_true", help="Não roda a comparação com o 7B.")
    return p.parse_args()

# ----------------------------------------------------------------------------- chaves S3 do corpus
MANIFEST_KEY = "corpus/raw/v1_refseq/composed/composition_manifest.parquet"
FEATURES_KEY = "corpus/raw/v1_refseq/features/genome_features.parquet"
FASTA_KEY    = "corpus/raw/v1_refseq/fasta/all_genomes.fasta"
GBFF_KEY     = "corpus/raw/v1_refseq/genbank/viral.1.genomic.gbff.gz"
# caches model-independent / por-modelo no S3 (sob experiments/embed_probes/)
ENC_CACHE_S3 = "experiments/embed_probes/features/codon_enc_cds.parquet"
EMB7B_BALT_S3 = "experiments/embed_probes/embeddings/embeddings_baltimore_evo2_7b_w32768.npz"
EMB7B_FEAT_S3 = "experiments/embed_probes/embeddings/embeddings_features_evo2_7b_w32768.npz"

BALT_NAMES = {"I":"I dsDNA","II":"II ssDNA","III":"III dsRNA","IV":"IV ssRNA(+)",
              "V":"V ssRNA(-)","VI":"VI ssRNA-RT","VII":"VII dsDNA-RT"}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ----------------------------------------------------------------------------- helpers de sequência
_COMP = str.maketrans("ACGT", "TGCA")
def _rc(s): return s.translate(_COMP)[::-1]
_SYN = {
 2:[["TTT","TTC"],["TAT","TAC"],["CAT","CAC"],["CAA","CAG"],["AAT","AAC"],
    ["AAA","AAG"],["GAT","GAC"],["GAA","GAG"],["TGT","TGC"]],
 3:[["ATT","ATC","ATA"]],
 4:[["GTT","GTC","GTA","GTG"],["CCT","CCC","CCA","CCG"],["ACT","ACC","ACA","ACG"],
    ["GCT","GCC","GCA","GCG"],["GGT","GGC","GGA","GGG"]],
 6:[["TTA","TTG","CTT","CTC","CTA","CTG"],["CGT","CGC","CGA","CGG","AGA","AGG"],
    ["TCT","TCC","TCA","TCG","AGT","AGC"]],
}
def enc_from_counter(cc):
    import numpy as np
    Fcls = {}
    for deg, fams in _SYN.items():
        Fs = []
        for fam in fams:
            n = sum(cc.get(c,0) for c in fam)
            if n < 2: continue
            p2 = sum((cc.get(c,0)/n)**2 for c in fam)
            F = (n*p2 - 1)/(n - 1)
            if F > 0: Fs.append(F)
        if Fs: Fcls[deg] = sum(Fs)/len(Fs)
    if 2 not in Fcls and 4 not in Fcls: return np.nan
    F2 = Fcls.get(2); F4 = Fcls.get(4)
    if F2 is None: F2 = F4
    if F4 is None: F4 = F2
    F3 = Fcls.get(3, (F2+F4)/2); F6 = Fcls.get(6, F4)
    return float(min(61.0, max(20.0, 2 + 9/F2 + 1/F3 + 5/F4 + 3/F6)))
def codon_enc_genome(seq):
    cc = Counter(seq[i:i+3] for i in range(0, len(seq)-2, 3))
    cc = Counter({k:v for k,v in cc.items() if all(b in "ACGT" for b in k)})
    return enc_from_counter(cc)
def gc_regional_std(seq, win=500):
    import numpy as np
    n = len(seq)//win
    if n < 2: return 0.0
    gcs = [(seq.count("G",i*win,(i+1)*win)+seq.count("C",i*win,(i+1)*win))/win for i in range(n)]
    return float(np.std(gcs))
_MOTIFS = ["TAATATTAC","GCTGGTGG"]   # nonanucleotídeo CRESS + sítio Chi
def recomb_ir_density(seq, k=6):
    L=len(seq)
    if L < k: return 0.0
    ir = sum(1 for i in range(L-k+1) if (sub:=seq[i:i+k])==_rc(sub))
    mt = sum(seq.count(m)+seq.count(_rc(m)) for m in _MOTIFS)
    return (ir+mt)/(L/1000.0)

# ----------------------------------------------------------------------------- setup de ambiente
def ensure_deps():
    def pip(*p): subprocess.run([sys.executable,"-m","pip","install","-q",*p], check=False)
    try:
        import sklearn, pandas, pyarrow, boto3, numpy  # noqa
    except Exception:
        pip("scikit-learn","pandas","pyarrow","boto3","numpy")

# ----------------------------------------------------------------------------- FP8 capability-aware
def patch_fp8(model_name):
    """Em Hopper/Ada (cc>=8.9) mantém FP8 LIGADO (H100 = mais rápido); em GPU < 8.9 desliga
    interceptando a leitura da config do Evo2 (pkgutil.get_data). Idempotente."""
    import yaml, torch
    import evo2.models as M
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
    want_fp8 = cc >= (8,9)
    log(f"GPU cc {cc[0]}.{cc[1]} -> use_fp8_input_projections = {want_fp8}")
    res = M.CONFIG_MAP[model_name]
    cfg = yaml.safe_load(M.pkgutil.get_data(M.__name__, res))
    cfg["use_fp8_input_projections"] = bool(want_fp8)
    patched = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    if not hasattr(M, "_orig_get_data"): M._orig_get_data = M.pkgutil.get_data
    _orig = M._orig_get_data
    def _gd(pkg, r): return patched if (pkg==M.__name__ and r==res) else _orig(pkg, r)
    M.pkgutil.get_data = _gd
    return want_fp8

# ----------------------------------------------------------------------------- embeddings
def make_windows(seq, window, stride, max_windows):
    import numpy as np
    L = len(seq)
    if L <= window: return [seq]
    starts = list(range(0, L - window + 1, stride))
    if starts[-1] != L - window: starts.append(L - window)
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts)-1, max_windows).round().astype(int)
        starts = sorted(set(starts[j] for j in idx))
    return [seq[s:s+window] for s in starts]

def build_embedder(model, layer, device, window, stride, max_windows):
    import numpy as np, torch
    @torch.inference_mode()
    def embed(seq):
        seq = seq.upper(); vecs = []
        for w in make_windows(seq, window, stride, max_windows):
            ids = torch.tensor(model.tokenizer.tokenize(w), dtype=torch.int).unsqueeze(0).to(device)
            _, emb = model(ids, return_embeddings=True, layer_names=[layer])
            vecs.append(emb[layer].float().mean(dim=1).squeeze(0).cpu().numpy())
            del ids, emb
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return np.mean(vecs, axis=0)
    return embed

# ----------------------------------------------------------------------------- probes (reusáveis 7B/20B)
def cv_r2(Xm, yv, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, KFold
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1,1,10,100,1000]))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, Xm, yv, cv=cv, scoring="r2").mean())

def probe_baltimore(X, y, gc, length, seed):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    yp = cross_val_predict(clf, X, y, cv=cv)
    acc = accuracy_score(y, yp); f1 = f1_score(y, yp, average="macro")
    Xtriv = np.column_stack([gc, length])
    yp_t = cross_val_predict(clf, Xtriv, y, cv=cv)
    acc_t = accuracy_score(y, yp_t)
    return dict(acc=round(acc,3), f1=round(f1,3), acc_gc_len=round(acc_t,3), gain=round(acc-acc_t,3))

def probe_features(X, feat_df, accs, seed, targets):
    import numpy as np
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
    import numpy as np, pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    m = pd.Series(labels).isin(valid).values
    if m.sum() < 50: return None
    Xc, yc = X[m], np.asarray(labels)[m]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    yp = cross_val_predict(clf, Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return dict(n=int(m.sum()), n_classes=len(set(yc.tolist())),
                acc=round(accuracy_score(yc,yp),3), f1=round(f1_score(yc,yp,average="macro"),3))

# ----------------------------------------------------------------------------- codon CDS (gbff)
def load_codon_cds(s3, bucket, local_dir, enc_cache, accs, skip, no_upload):
    import numpy as np, pandas as pd
    if skip:
        log("codon_enc_cds: pulado (--skip-codon-cds)"); return None
    if os.path.exists(enc_cache):
        log("codon_enc_cds: cache local"); return pd.read_parquet(enc_cache)
    try:
        s3.download_file(bucket, ENC_CACHE_S3, enc_cache)
        log("codon_enc_cds: cache S3"); return pd.read_parquet(enc_cache)
    except Exception:
        pass
    # precisa varrer o gbff (~450MB) — model-independent, roda 1x no projeto inteiro
    from Bio import SeqIO
    gbff = f"{local_dir}/viral.1.genomic.gbff.gz"
    if not (os.path.exists(gbff) and os.path.getsize(gbff) > 0):
        log("baixando gbff (~450 MB)..."); s3.download_file(bucket, GBFF_KEY, gbff)
    wanted = set(accs.tolist()); base2acc = {a.split(".")[0]: a for a in accs}
    rows, seen = [], set()
    with gzip.open(gbff, "rt") as fh:
        for rec in SeqIO.parse(fh, "genbank"):
            acc = rec.id if rec.id in wanted else base2acc.get((rec.id or "").split(".")[0]) or base2acc.get(rec.name)
            if acc is None or acc in seen: continue
            seen.add(acc); cc = Counter()
            for f in rec.features:
                if f.type != "CDS" or "pseudo" in f.qualifiers: continue
                try: cds = str(f.extract(rec.seq)).upper()
                except Exception: continue
                cs = int(f.qualifiers.get("codon_start", ["1"])[0]) - 1
                cds = cds[cs:]
                for i in range(0, len(cds)-2, 3):
                    cod = cds[i:i+3]
                    if cod[0] in "ACGT" and cod[1] in "ACGT" and cod[2] in "ACGT": cc[cod]+=1
            ncod = sum(cc.values())
            rows.append((acc, enc_from_counter(cc) if ncod>=30 else np.nan, ncod))
            if len(seen) >= len(wanted): break
    df = pd.DataFrame(rows, columns=["accession","codon_enc_cds","n_codons_cds"]).set_index("accession")
    df.to_parquet(enc_cache)
    if not no_upload:
        try: s3.upload_file(enc_cache, bucket, ENC_CACHE_S3); log("codon_enc_cds salvo no S3")
        except Exception as e: log("aviso: upload codon_enc_cds falhou:", e)
    return df

# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    ensure_deps()
    import numpy as np, pandas as pd, boto3
    os.makedirs(args.local_dir, exist_ok=True)
    os.makedirs(f"{args.local_dir}/results", exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    tag = args.model
    emb_cache = f"{args.local_dir}/embeddings_{tag}_w{args.window}.npz"
    emb_cache_s3 = f"{args.out_s3.rstrip('/')}/embeddings/embeddings_{tag}_w{args.window}.npz"

    # ---- 1. dados do corpus
    for key in (MANIFEST_KEY, FEATURES_KEY, FASTA_KEY):
        dst = f"{args.local_dir}/{os.path.basename(key)}"
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            log("baixando", key); s3.download_file(args.bucket, key, dst)
    man  = pd.read_parquet(f"{args.local_dir}/{os.path.basename(MANIFEST_KEY)}")
    feat = pd.read_parquet(f"{args.local_dir}/{os.path.basename(FEATURES_KEY)}")
    feat_cols = [c for c in ["accession","GC","genome_length","coding_fraction","gene_density",
                 "n_genes","n_CDS","noncoding_bp","mean_intergenic_len"] if c in feat.columns and (c=="accession" or c not in man.columns)]
    meta = man.merge(feat[feat_cols], on="accession", how="left")

    # ---- 2. definir o conjunto de accessions (reuso do 7B = comparação limpa)
    balt_accs = feat_accs = None; balt_y = None; X7b_balt = X7b_feat = None
    if args.reuse_7b_accs:
        try:
            bdst = f"{args.local_dir}/{os.path.basename(EMB7B_BALT_S3)}"
            fdst = f"{args.local_dir}/{os.path.basename(EMB7B_FEAT_S3)}"
            if not os.path.exists(bdst): s3.download_file(args.bucket, EMB7B_BALT_S3, bdst)
            if not os.path.exists(fdst): s3.download_file(args.bucket, EMB7B_FEAT_S3, fdst)
            db = np.load(bdst, allow_pickle=True); X7b_balt = db["X"].astype(np.float32)
            balt_accs = db["accs"].astype(str); balt_y = db["y"].astype(str)
            dfe = np.load(fdst, allow_pickle=True); X7b_feat = dfe["X"].astype(np.float32)
            feat_accs = dfe["accs"].astype(str)
            log(f"reuso 7B: baltimore {len(balt_accs)} accs, features {len(feat_accs)} accs")
        except Exception as e:
            log("aviso: não consegui reusar accs do 7B (", e, ") -> fresh sample"); args.reuse_7b_accs = False
    if not args.reuse_7b_accs:
        sub = (meta.dropna(subset=["coding_fraction","gene_density","genome_length","GC"])
                   .groupby("quota_group", group_keys=False)
                   .apply(lambda g: g.sample(min(args.n_per_group, len(g)), random_state=args.seed)))
        feat_accs = sub["accession"].to_numpy().astype(str)
        bsub = meta[meta["baltimore"].isin(BALT_NAMES)]
        balt_accs = bsub["accession"].to_numpy().astype(str)
        balt_y = bsub.set_index("accession").reindex(balt_accs)["baltimore"].to_numpy().astype(str)

    wanted = sorted(set(balt_accs.tolist()) | set(feat_accs.tolist()))
    if args.limit:
        wanted = wanted[:args.limit]
        wset = set(wanted)
        keepb = np.array([a in wset for a in balt_accs]); balt_accs, balt_y = balt_accs[keepb], balt_y[keepb]
        if X7b_balt is not None: X7b_balt = X7b_balt[keepb]
        keepf = np.array([a in wset for a in feat_accs]); feat_accs = feat_accs[keepf]
        if X7b_feat is not None: X7b_feat = X7b_feat[keepf]
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

    # ---- 4. embeddings do modelo-alvo (calcula 1x, cacheia)
    if os.path.exists(emb_cache):
        d = np.load(emb_cache, allow_pickle=True)
        emb = {a: v for a, v in zip(d["accs"].astype(str), d["X"].astype(np.float32))}
        log(f"embeddings {tag} do cache: {len(emb)}")
    else:
        log(f"carregando modelo {args.model} (local_path={args.weights_local}) ...")
        patch_fp8(args.model)
        gc.collect()
        from evo2 import Evo2
        import torch
        model = Evo2(args.model, local_path=args.weights_local) if args.weights_local else Evo2(args.model)
        model.model.eval()
        if torch.cuda.is_available():
            log(f"GPU {torch.cuda.get_device_name(0)} | VRAM pós-load "
                f"{torch.cuda.memory_allocated()/1e9:.1f} GB")
        embed = build_embedder(model, args.layer, args.device, args.window, args.stride, args.max_windows)
        emb = {}
        t0 = time.time()
        for i, a in enumerate(wanted):
            if a not in seqs: continue
            try:
                emb[a] = embed(seqs[a])
            except Exception as e:
                log("falhou", a, "->", repr(e))
            if (i+1) % 25 == 0:
                log(f"  {i+1}/{len(wanted)} | {(time.time()-t0)/(i+1):.2f}s/genoma")
        accs_arr = np.array(list(emb.keys()))
        X_arr = np.vstack([emb[a] for a in accs_arr]).astype(np.float32)
        np.savez_compressed(emb_cache, X=X_arr, accs=accs_arr)
        log(f"embeddings salvos: {emb_cache} {X_arr.shape}")
        if not args.no_upload:
            try: s3.upload_file(emb_cache, args.bucket, emb_cache_s3); log("embeddings -> S3", emb_cache_s3)
            except Exception as e: log("aviso: upload embeddings falhou:", e)
        del model
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

    # ---- 5. montar matrizes alinhadas + alvos
    def stack(accs_list):
        ok = [a for a in accs_list if a in emb]
        return np.vstack([emb[a] for a in ok]).astype(np.float32), np.array(ok)
    # features extras calculadas da sequência (model-independent)
    def clean(a): return "".join(c for c in seqs[a].upper() if c in "ACGT")
    enc_cds = load_codon_cds(s3, args.bucket, args.local_dir,
                             f"{args.local_dir}/codon_enc_cds.parquet",
                             np.array(wanted), args.skip_codon_cds, args.no_upload)

    results = {"model": tag, "layer": args.layer, "window": args.window,
               "n_baltimore": 0, "n_features": 0}

    # ---- 6. probe Baltimore
    Xb, ab = stack(balt_accs.tolist())
    yb = pd.Series(balt_y, index=balt_accs).reindex(ab).to_numpy().astype(str)
    fb = meta.set_index("accession").reindex(ab)
    res_balt = probe_baltimore(Xb, yb, fb["GC"].fillna(0).values,
                               fb["genome_length"].fillna(0).values, args.seed)
    results["baltimore"] = res_balt; results["n_baltimore"] = len(ab)
    log("Baltimore:", res_balt)

    # ---- 7. probes de features
    Xf, af = stack(feat_accs.tolist())
    fdf = meta.set_index("accession").reindex(af).copy()
    fdf["codon_enc_genome"]  = [codon_enc_genome(clean(a)) for a in af]
    fdf["gc_regional_std"]   = [gc_regional_std(clean(a)) for a in af]
    fdf["recomb_ir_density"] = [recomb_ir_density(clean(a)) for a in af]
    if enc_cds is not None:
        fdf["codon_enc_cds"] = enc_cds["codon_enc_cds"].reindex(af).values
    targets = [("coding_fraction",False),("gene_density",False),("n_genes",True),
               ("noncoding_bp",True),("mean_intergenic_len",True),
               ("codon_enc_cds",False),("codon_enc_genome",False),
               ("gc_regional_std",False),("recomb_ir_density",True)]
    rows = probe_features(Xf, fdf, af, args.seed, targets)
    results["features"] = rows; results["n_features"] = len(af)
    log("Features (R²_inc):", {r["feature"]: r["R2_inc"] for r in rows})

    # ---- 8. host / família
    results["host"]   = probe_classification(Xf, fdf["host"].values, args.seed,
                                             {"eukaryote","bacteria","archaea"})
    fams = fdf["family"].replace("", np.nan).dropna().value_counts()
    top = fams[fams>=25].index.tolist()
    results["family"] = probe_classification(Xf, fdf["family"].values, args.seed, set(top)) if len(top)>=2 else None
    log("host:", results["host"], "| família:", results["family"])

    # ---- 9. comparação 7B vs 20B (mesmas accs)
    if not args.no_compare_7b and X7b_balt is not None and tag != "evo2_7b":
        try:
            # baltimore 7B sobre as MESMAS accs presentes em ab
            idxb = {a:i for i,a in enumerate(balt_accs)}
            sel = [idxb[a] for a in ab if a in idxb]
            r7b_balt = probe_baltimore(X7b_balt[sel], yb,
                                       fb["GC"].fillna(0).values, fb["genome_length"].fillna(0).values, args.seed)
            idxf = {a:i for i,a in enumerate(feat_accs)}
            self_idx = [idxf[a] for a in af if a in idxf]
            r7b_feat = probe_features(X7b_feat[self_idx], fdf, af, args.seed, targets)
            results["compare_7b"] = {
                "baltimore_7b": r7b_balt, "baltimore_20b": res_balt,
                "delta_baltimore_acc": round(res_balt["acc"]-r7b_balt["acc"], 3),
                "features_7b": {r["feature"]: r["R2_inc"] for r in r7b_feat},
                "features_20b": {r["feature"]: r["R2_inc"] for r in rows},
            }
            log("Δ Baltimore acc (20B - 7B):", results["compare_7b"]["delta_baltimore_acc"])
        except Exception as e:
            log("aviso: comparação 7B falhou:", e)

    # ---- 10. salvar resultados
    out_json = f"{args.local_dir}/results/probe_results_{tag}.json"
    with open(out_json, "w") as f: json.dump(results, f, indent=2, default=str)
    pd.DataFrame(rows).to_csv(f"{args.local_dir}/results/features_{tag}.csv", index=False)
    log("resultados:", out_json)
    if not args.no_upload:
        try:
            s3.upload_file(out_json, args.bucket, f"{args.out_s3.rstrip('/')}/results/probe_results_{tag}.json")
            s3.upload_file(f"{args.local_dir}/results/features_{tag}.csv", args.bucket,
                           f"{args.out_s3.rstrip('/')}/results/features_{tag}.csv")
            log("resultados -> S3", f"{args.out_s3.rstrip('/')}/results/")
        except Exception as e: log("aviso: upload resultados falhou:", e)

    # ---- resumo legível
    print("\n" + "="*64 + f"\nRESUMO — {tag}\n" + "="*64)
    print(f"Baltimore: acc {res_balt['acc']} (GC+len {res_balt['acc_gc_len']}, ganho {res_balt['gain']:+}) "
          f"| n={results['n_baltimore']}")
    print(f"{'feature':<22}{'R2_inc':>8}{'R2_emb':>8}{'n':>7}")
    for r in sorted(rows, key=lambda x:-x["R2_inc"]):
        print(f"{r['feature']:<22}{r['R2_inc']:>8}{r['R2_emb']:>8}{r['n']:>7}")
    if results.get("host"):   print("host  :", results["host"])
    if results.get("family"): print("família:", results["family"])
    if results.get("compare_7b"):
        c = results["compare_7b"]; print(f"\nΔ vs 7B — Baltimore acc: {c['delta_baltimore_acc']:+}")
        for fkey in c["features_20b"]:
            d = round(c["features_20b"][fkey]-c["features_7b"].get(fkey, float('nan')), 3)
            print(f"   {fkey:<22} 7B {c['features_7b'].get(fkey)} -> 20B {c['features_20b'][fkey]}  (Δ {d:+})")
    print("="*64)

if __name__ == "__main__":
    main()
