#!/usr/bin/env python3
"""Artefatos pequenos p/ figuras do 20B: confusion matrix (Baltimore, StratifiedKFold 5,
mesma metodologia do figure1 original) + PCA(2) do embedding (blocks.18) colorido por
coding_fraction. Roda onde os embeddings estiverem cacheados; salva só JSON pequeno."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_analysis import load_meta, load_7b, load_20b, align, groups, OUTDIR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

SEED = 42
BALT = ["I","II","III","IV","V","VI","VII"]

def main():
    Xb_7b, accb = load_7b("baltimore")
    X20, ab20 = load_20b(18, "fp8")
    Xb_al, kb = align(X20, ab20, accb)
    assert kb.all(), "cobertura incompleta"
    db = load_meta(accb)
    yb = db["baltimore"].values

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced"))
    yp = cross_val_predict(pipe, Xb_al, yb, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    labels = [b for b in BALT if b in set(yb)]
    cm = confusion_matrix(yb, yp, labels=labels, normalize="true")
    acc = accuracy_score(yb, yp)

    X7b_f, accf = load_7b("features")
    X20f, af20 = load_20b(18, "fp8")
    Xf_al, kf = align(X20f, af20, accf)
    assert kf.all()
    dff = load_meta(accf)
    Z = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(Xf_al))

    out = {
        "cm": {"labels": labels, "M": cm.tolist(), "acc": float(acc), "n": int(len(yb))},
        "pca": {"PC1": Z[:,0].tolist(), "PC2": Z[:,1].tolist(),
                "coding_fraction": dff["coding_fraction"].values.astype(float).tolist(),
                "n": int(len(Z))},
    }
    json.dump(out, open(f"{OUTDIR}/fig_artifacts_20b.json", "w"))
    print("[ok] fig_artifacts_20b.json  acc=", acc, "n_pca=", len(Z))

if __name__ == "__main__":
    main()
