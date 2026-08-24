#!/usr/bin/env python3
"""classification_metrics.py — beyond accuracy (R2 #15).

R2 #15 notes that accuracy alone is insufficient for the family classification task, and asks
for the number of families included, macro-F1, balanced accuracy and per-family recall.

Reports, for Baltimore class, host domain and family: accuracy, balanced accuracy, macro-F1,
weighted F1, and per-class recall with the n of each class.

Design note: for **family** the `family` scheme does not exist — one cannot predict a family
never seen in training. Family is reported under the pre-registered `cl95` scheme only. For
Baltimore class and host domain both schemes are reported.

Usage:
    PYTHONPATH=<repo>/code/03_analysis python classification_metrics.py --out ../../results/json
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
    """Out-of-fold predictions with groups preserved. Stacks the repeats."""
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
    fam = md["family"].fillna("<no family>").values.astype(str)

    out = {"config": {"splits": SPLITS, "repeats": REPEATS, "seed": SEED,
                      "min_family_n": MIN_FAMILY_N,
                      "note_family_scheme":
                          "Family is not evaluated under family grouping: one cannot predict "
                          "a class absent from training. The pre-registered cl95 scheme only."},
           "targets": {}}

    # The populations must be the SAME as in the paper, otherwise the new metrics are not
    # comparable with the published table. Baltimore uses the 981-record subset; host domain
    # and family use the 1,200-record one. Without this, the 240 records with baltimore "?"
    # enter as an eighth class and n becomes 1,912, which is what an earlier version did.
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
    # Family: feature subset, well-represented families only, and cl95 only
    vc = md.loc[in_feat, "family"].value_counts()
    keep_f = vc[vc >= MIN_FAMILY_N].index
    m = in_feat & md["family"].isin(keep_f).values
    tasks.append(("family", m, md["family"].values, ["cl95"]))
    print(f"populations: baltimore={tasks[0][1].sum()} host={tasks[1][1].sum()} "
          f"family={tasks[2][1].sum()} ({len(keep_f)} families >= {MIN_FAMILY_N})")

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

    # per-family recall, sorted: what R2 #15 asks for by name
    f = out["targets"]["family"]["reps"]["evo2_20b_blocks18"]["cl95"]["per_class_recall"]
    print(f"\n--- per-family recall (Evo 2, cl95; {len(f)} families) ---")
    for k, v in sorted(f.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"  {k:<24} recall={v['recall']:.3f}  n={v['n']}")

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "classification_metrics.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
