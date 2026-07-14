#!/usr/bin/env python3
"""Cluster-aware CV (#1): re-roda os probes com StratifiedGroupKFold/GroupKFold,
agrupando por cluster de identidade (MMseqs2 95%), e compara com a CV aleatória
cacheada. Mede o efeito de leakage de quase-duplicatas."""
import os, pickle, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "..")
# Embeddings (não versionados; ver data/README.md) e inputs do repo.
NB    = os.environ.get("SCALE_EMB",  os.path.join(REPO, "embeddings"))
DATA  = os.environ.get("SCALE_DATA", os.path.join(REPO, "data"))
# Métricas do 7B sob CV ALEATÓRIA — o comparador contra o qual medimos o efeito de leakage.
CACHE = os.path.join(REPO, "results", "probe_cache_7b_randomcv.pkl")
SEED  = 42
RND=pickle.load(open(CACHE,"rb"))

# mapa acc->cluster (95%)
cmap={}
for ln in open(f"{DATA}/cl95_cluster.tsv"):
    rep,mem=ln.rstrip("\n").split("\t"); cmap[mem]=rep
def groups(accs): return np.array([cmap.get(a,a) for a in accs])

def load(f): d=np.load(f"{NB}/{f}",allow_pickle=True); return d["X"],d["accs"]
def meta(accs):
    man=pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    fe=pd.read_parquet(f"{DATA}/features.parquet").set_index("accession")
    df=pd.DataFrame(index=accs)
    for c in ["host","family","baltimore"]: df[c]=man.reindex(accs)[c].values
    for c in ["GC","genome_length","coding_fraction","gene_density","noncoding_bp","n_genes","mean_intergenic_len"]:
        df[c]=fe.reindex(accs)[c].values
    return df

def clf_group(X,y,g,reps=3):
    sc=[]
    for r in range(reps):
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+r)
        sc+=list(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,class_weight="balanced")),X,y,cv=cv.split(X,y,g),scoring="accuracy"))
    return np.array(sc)
def reg_group(X,y,g,reps=3):
    sc=[]
    rng=np.random.default_rng(SEED)
    uniq=np.unique(g)
    for r in range(reps):
        perm={u:i for i,u in enumerate(rng.permutation(uniq))}
        fold=np.array([perm[x]%5 for x in g])
        for k in range(5):
            tr,te=fold!=k,fold==k
            if te.sum()<2 or len(np.unique(te))<1: continue
            m=make_pipeline(StandardScaler(),RidgeCV(alphas=[1,10,100,1000])).fit(X[tr],y[tr])
            from sklearn.metrics import r2_score
            sc.append(r2_score(y[te],m.predict(X[te])))
    return np.array(sc)

print(f"{'Task':22s} {'random CV':>16s} {'cluster CV (95%)':>18s} {'Δ':>7s}")
Xb,ab=load("embeddings_baltimore_evo2_7b_w32768.npz"); db=meta(ab); gb=groups(ab)
e=clf_group(Xb,db['baltimore'].values,gb); r=RND['clf']['Baltimore']['emb']
print(f"{'Baltimore (acc)':22s} {r.mean():.3f}±{r.std():.3f}   {e.mean():.3f}±{e.std():.3f}   {e.mean()-r.mean():+.3f}")
Xf,af=load("embeddings_features_evo2_7b_w32768.npz"); dff=meta(af); gf=groups(af)
hm=dff['host'].isin(['eukaryote','bacteria','archaea']).values
e=clf_group(Xf[hm],dff['host'].values[hm],gf[hm]); r=RND['clf']['Host domain']['emb']
print(f"{'Host (acc)':22s} {r.mean():.3f}±{r.std():.3f}   {e.mean():.3f}±{e.std():.3f}   {e.mean()-r.mean():+.3f}")
fam=dff['family'].replace('',np.nan); top=fam.value_counts(); top=top[top>=25].index.tolist()
fk=dff['family'].isin(top).values
e=clf_group(Xf[fk],dff['family'].values[fk],gf[fk]); r=RND['clf']['Family']['emb']
print(f"{'Family (acc)':22s} {r.mean():.3f}±{r.std():.3f}   {e.mean():.3f}±{e.std():.3f}   {e.mean()-r.mean():+.3f}")
for f,logt in [("coding_fraction",False),("gene_density",False),("noncoding_bp",True),("n_genes",True),("mean_intergenic_len",True)]:
    y=dff[f].values.astype(float); y=np.log1p(np.clip(y,0,None)) if logt else y
    e=reg_group(Xf,y,gf); r=RND['reg'][f]['emb']
    print(f"{f+' (R2)':22s} {r.mean():.3f}±{r.std():.3f}   {e.mean():.3f}±{e.std():.3f}   {e.mean()-r.mean():+.3f}")
