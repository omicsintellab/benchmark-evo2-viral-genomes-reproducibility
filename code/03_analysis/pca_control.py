#!/usr/bin/env python3
"""pca_control.py — Controle de dimensionalidade p/ a comparação de escala 7B x 20B.

Problema: o embedding do 20B tem 8192 dims (vs 4096 no 7B e no baseline 6-mer).
Um ganho do 20B em taxonomia (Results 3.4) pode ser efeito de ESCALA (mais
parâmetros -> representação melhor) ou artefato de DIMENSIONALIDADE (mais graus
de liberdade no probe linear, mesmo n). Este script separa os dois: reduz cada
camada do 20B a 4096 componentes via PCA (fit DENTRO da pipeline/fold — sem
vazamento) e re-roda a mesma suíte de probes. Se o ganho sobrevive à redução ->
efeito real de escala; se desaparece -> era dimensionalidade.

Reusa os helpers de scale_analysis.py (mesmos dados/paths via env vars).
Roda em CPU; independente do scale_analysis.py (não precisa esperar ele acabar,
mas evite rodar os dois ao mesmo tempo pra não competir pelos mesmos núcleos).

Nota sobre PCA_DIM: NÃO dá pra usar 4096 componentes (o "casamento" ingênuo com o
7B) — PCA tem no máximo min(n_amostras_treino, n_features) componentes, e o alvo
mais restritivo (family, n=349, ~4/5 em treino por fold ≈ 279) já é bem menor que
4096. Usamos um n_components que cabe em TODOS os alvos com folga (< 279) e
reduzimos o 7B/6-mer ao MESMO valor para a comparação de dimensionalidade
igualada ser justa dos dois lados, não só uma poda do 20B.

Uso: python pca_control.py [--layers 15,18] (default: LAYERS_20B do scale_analysis)
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_analysis import (load_meta, load_7b, load_20b, align, groups,
        clf_rand, reg_rand, clf_group, reg_group, REG_FEATS, LAYERS_20B, OUTDIR)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold,
        RepeatedKFold, StratifiedGroupKFold)
from sklearn.metrics import r2_score

SEED, SPLITS, REPEATS = 42, 5, 3
# Family é o alvo mais restritivo (n=349, treino/fold ~279 amostras) — PCA_DIM
# precisa ficar bem abaixo disso em TODOS os folds/repeats. 150 dá folga segura
# (StratifiedGroupKFold pode desbalancear um pouco os folds).
PCA_DIM = 150

def clf_rand_pca(X, y):
    cv = RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=PCA_DIM, random_state=SEED),
                         LogisticRegression(max_iter=1500, class_weight="balanced"))
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

def reg_rand_pca(X, y):
    cv = RepeatedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=PCA_DIM, random_state=SEED),
                         RidgeCV(alphas=[1, 10, 100, 1000]))
    return cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)

def clf_group_pca(X, y, g, reps=REPEATS):
    sc = []
    for r in range(reps):
        cv = StratifiedGroupKFold(n_splits=SPLITS, shuffle=True, random_state=SEED+r)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=PCA_DIM, random_state=SEED),
                             LogisticRegression(max_iter=1500, class_weight="balanced"))
        sc += list(cross_val_score(pipe, X, y, cv=cv.split(X, y, g), scoring="accuracy", n_jobs=-1))
    return np.array(sc)

def reg_group_pca(X, y, g, reps=REPEATS):
    sc = []; rng = np.random.default_rng(SEED); uniq = np.unique(g)
    for r in range(reps):
        perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
        fold = np.array([perm[x] % SPLITS for x in g])
        for k in range(SPLITS):
            tr, te = fold != k, fold == k
            if te.sum() < 2: continue
            pipe = make_pipeline(StandardScaler(), PCA(n_components=PCA_DIM, random_state=SEED),
                                 RidgeCV(alphas=[1, 10, 100, 1000])).fit(X[tr], y[tr])
            sc.append(r2_score(y[te], pipe.predict(X[te])))
    return np.array(sc)

def suite_clf_pca(X, y, g):
    return {"rand_pca": clf_rand_pca(X, y), "clus_pca": clf_group_pca(X, y, g)}
def suite_reg_pca(X, y, g):
    return {"rand_pca": reg_rand_pca(X, y), "clus_pca": reg_group_pca(X, y, g)}

def main():
    layers = LAYERS_20B
    for a in sys.argv[1:]:
        if a.startswith("--layers="):
            layers = [int(x) for x in a.split("=", 1)[1].split(",")]

    X7b_b, accb = load_7b("baltimore")
    X7b_f, accf = load_7b("features")
    db, dff = load_meta(accb), load_meta(accf)
    gb, gf = groups(accb), groups(accf)
    hm = dff["host"].isin(["eukaryote", "bacteria", "archaea"]).values
    fam = dff["family"].replace("", np.nan); topf = fam.value_counts(); topf = topf[topf >= 25].index.tolist()
    fk = dff["family"].isin(topf).values

    def run_rep(name, Xb, Xf):
        print(f"[pca_control] {name} (dim={Xb.shape[1]} -> {PCA_DIM}) ...", flush=True)
        r = {}
        r["Baltimore"] = suite_clf_pca(Xb, db["baltimore"].values, gb)
        r["Host"] = suite_clf_pca(Xf[hm], dff["host"].values[hm], gf[hm])
        r["Family"] = suite_clf_pca(Xf[fk], dff["family"].values[fk], gf[fk])
        for f, logt in REG_FEATS:
            y = dff[f].values.astype(float); y = np.log1p(np.clip(y, 0, None)) if logt else y
            r[f] = suite_reg_pca(Xf, y, gf)
        print(f"  Baltimore (clus_pca): {r['Baltimore']['clus_pca'].mean():.3f}", flush=True)
        return r

    out = {"pca_dim": PCA_DIM, "layers": layers, "reps": {}}
    # 7B reduzido ao mesmo PCA_DIM — referência para a comparação igualada
    out["reps"]["7B_blocks28"] = run_rep("7B_blocks28", X7b_b, X7b_f)
    for L in layers:
        Xb20, ab20 = load_20b(L, "fp8"); Xf20, af20 = load_20b(L, "fp8")
        if Xb20 is None:
            print(f"[aviso] L{L} fp8 ausente, pulando"); continue
        Xb_al, kb = align(Xb20, ab20, accb); Xf_al, kf = align(Xf20, af20, accf)
        if not (kb.all() and kf.all()):
            print(f"[aviso] L{L}: cobertura incompleta, pulando"); continue
        out["reps"][f"20B_blocks{L}"] = run_rep(f"20B_blocks{L}", Xb_al, Xf_al)

    def dump(d):
        if isinstance(d, dict): return {k: dump(v) for k, v in d.items()}
        if isinstance(d, np.ndarray): return {"mean": float(d.mean()), "std": float(d.std())}
        return d
    json.dump(dump(out), open(f"{OUTDIR}/pca_control_metrics.json", "w"), indent=2)
    print("[ok] pca_control_metrics.json", flush=True)

if __name__ == "__main__":
    main()
