#!/usr/bin/env python3
"""classification_metrics.py — além da acurácia (R2 #15).

R2 #15: "acurácia sozinha é insuficiente para a tarefa de classificação de família. O número
de famílias incluídas, o macro-F1, a acurácia balanceada e o recall por família devem ser
reportados."

Reporta, para Baltimore, hospedeiro e família: acurácia, acurácia balanceada, macro-F1,
F1 ponderado, e recall por classe, com o n de cada classe.

Nota de desenho: para **família** o esquema `family` não existe — não se prevê uma família
que nunca apareceu no treino. Família sai só sob o esquema pré-registrado `cl95`. Para
Baltimore e hospedeiro os dois esquemas são reportados.

Uso:
    PYTHONPATH=<repro>/code/03_analysis python classification_metrics.py --out ../../results/json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scale_analysis import DATA, load_20b, kmer_matrix, groups as cl95_groups
from family_cv import SPLITS, REPEATS, SEED

MIN_FAMILY_N = 25          # mesmo corte usado no artigo (precision_control.py)


def clf_pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1500, class_weight="balanced"))


def oof_predictions(X, y, g, reps=REPEATS):
    """Predições out-of-fold com grupos preservados. Empilha as repetições."""
    rng = np.random.default_rng(SEED)
    yt, yp = [], []
    uniq = np.unique(g)
    for r in range(reps):
        perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
        fold = np.array([perm[x] % SPLITS for x in g])
        for k in range(SPLITS):
            tr, te = fold != k, fold == k
            if te.sum() < 2 or len(np.unique(y[tr])) < 2:
                continue
            m = clf_pipe().fit(X[tr], y[tr])
            yt.append(y[te]); yp.append(m.predict(X[te]))
    return np.concatenate(yt), np.concatenate(yp)


def metrics(yt, yp):
    labs = sorted(set(yt))
    rec = recall_score(yt, yp, labels=labs, average=None, zero_division=0)
    return {"n_classes": len(labs), "n_obs": int(len(yt)),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "per_class_recall": {str(l): {"recall": float(r),
                                          "n": int((np.asarray(yt) == l).sum())}
                                 for l, r in zip(labs, rec)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "json"))
    ap.add_argument("--subset-baltimore", required=True,
                    help="data/probe_subset_baltimore.tsv do repo de reprodutibilidade.")
    ap.add_argument("--subset-features", required=True,
                    help="data/probe_subset_features.tsv do repo de reprodutibilidade.")
    a = ap.parse_args()

    X20, acc20 = load_20b(18, "fp8")
    accs = acc20.astype(str)
    man = pd.read_parquet(f"{DATA}/manifest.parquet").set_index("accession")
    md = pd.DataFrame(index=accs)
    for c in ("family", "baltimore", "host"):
        md[c] = man.reindex(accs)[c].replace("", np.nan).values
    X20 = X20.astype(np.float32)
    KM = kmer_matrix(list(accs))
    g95 = cl95_groups(accs)
    fam = md["family"].fillna("<sem familia>").values.astype(str)

    out = {"config": {"splits": SPLITS, "repeats": REPEATS, "seed": SEED,
                      "min_family_n": MIN_FAMILY_N,
                      "note_family_scheme":
                          "Familia nao e avaliada sob agrupamento por familia: nao se preve "
                          "uma classe ausente do treino. So o esquema pre-registrado cl95."},
           "targets": {}}

    # As populações têm de ser as MESMAS do artigo, senão as métricas novas não são
    # comparáveis à Tabela publicada. Baltimore usa o subconjunto de 981; hospedeiro e
    # família usam o de 1.200. Sem isso, os 240 records com baltimore "?" entram como uma
    # oitava classe e o n vira 1.912 — foi o que aconteceu na primeira versão deste script.
    balt_accs = set(pd.read_csv(a.subset_baltimore, sep="\t")["accession"].astype(str))
    feat_accs = set(pd.read_csv(a.subset_features, sep="\t")["accession"].astype(str))
    in_balt = np.array([x in balt_accs for x in accs])
    in_feat = np.array([x in feat_accs for x in accs])
    VALID_BALT = {"I", "II", "III", "IV", "V", "VI", "VII"}

    tasks = []
    m = in_balt & md["baltimore"].isin(VALID_BALT).values
    tasks.append(("baltimore", m, md["baltimore"].values, ["cl95", "family"]))
    m = in_feat & md["host"].isin(["eukaryote", "bacteria", "archaea"]).values
    tasks.append(("host", m, md["host"].values, ["cl95", "family"]))
    # Família: subconjunto de features, só as bem representadas, e só cl95
    vc = md.loc[in_feat, "family"].value_counts()
    keep_f = vc[vc >= MIN_FAMILY_N].index
    m = in_feat & md["family"].isin(keep_f).values
    tasks.append(("family", m, md["family"].values, ["cl95"]))
    print(f"populações: baltimore={tasks[0][1].sum()} host={tasks[1][1].sum()} "
          f"family={tasks[2][1].sum()} ({len(keep_f)} famílias >= {MIN_FAMILY_N})")

    for name, mask, yall, schemes in tasks:
        y = np.asarray(yall)[mask].astype(str)
        rec = {"n": int(mask.sum()), "reps": {}}
        if name == "family":
            rec["families_included"] = int(len(keep_f))
            rec["families_excluded_below_min"] = int((vc < MIN_FAMILY_N).sum())
        print(f"\n=== {name} (n={mask.sum()}, {len(set(y))} classes) ===")
        for rep_name, R in (("evo2_20b_blocks18", X20), ("6mer", KM)):
            rec["reps"][rep_name] = {}
            for sch in schemes:
                g = (g95 if sch == "cl95" else fam)[mask]
                yt, yp = oof_predictions(R[mask], y, g)
                mm = metrics(yt, yp)
                rec["reps"][rep_name][sch] = mm
                print(f"  {rep_name:>20} {sch:>7}: acc={mm['accuracy']:.3f} "
                      f"bal_acc={mm['balanced_accuracy']:.3f} macroF1={mm['macro_f1']:.3f}")
        out["targets"][name] = rec

    # recall por família, ordenado — é o que R2 #15 pede nominalmente
    f = out["targets"]["family"]["reps"]["evo2_20b_blocks18"]["cl95"]["per_class_recall"]
    print(f"\n--- recall por família (Evo 2, cl95; {len(f)} famílias) ---")
    for k, v in sorted(f.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"  {k:<24} recall={v['recall']:.3f}  n={v['n']}")

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "classification_metrics.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
