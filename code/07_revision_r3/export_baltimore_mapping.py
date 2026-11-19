#!/usr/bin/env python3
"""export_baltimore_mapping.py — deposit the ICTV-VMR-to-Baltimore mapping rule (reviewer point #9).

The ICTV VMR "Genome" field is not identical to Baltimore class. The rule actually applied is
the lookup table in code/01_corpus/compose_v1_groups.py; this script exports it, together with
the counts it produced in the analysed populations, so the mapping is auditable without reading
the code. Strings that match no key are left unassigned ("?"): that is where ambisense and
unsigned-ssRNA records end up. Unassigned records are excluded from the Baltimore probe and
retained in the feature subset, where they are summarised separately.

Usage:
    python code/07_revision_r3/export_baltimore_mapping.py
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "01_corpus"))
from compose_v1_groups import BALT, norm  # noqa: E402

OUT = os.path.join(REPO, "data", "baltimore_mapping.tsv")

NOTE = {
    "I": "dsDNA", "II": "ssDNA (either polarity, including (+/-))", "III": "dsRNA",
    "IV": "ssRNA(+)", "V": "ssRNA(-)", "VI": "ssRNA-RT", "VII": "dsDNA-RT",
    "?": "no key matched: ambisense genomes and records whose VMR field carries no polarity",
}


def main():
    man = pd.read_csv(os.path.join(REPO, "data", "corpus_manifest.tsv.gz"), sep="\t")
    feat = pd.read_csv(os.path.join(REPO, "data", "probe_subset_features.tsv"), sep="\t")
    balt = pd.read_csv(os.path.join(REPO, "data", "probe_subset_baltimore.tsv"), sep="\t")
    m = man.set_index("accession")
    n_corpus = m["baltimore"].fillna("?").value_counts()
    n_feat = m.reindex(feat.accession)["baltimore"].fillna("?").value_counts()
    n_balt = m.reindex(balt.accession)["baltimore"].fillna("?").value_counts()

    rows = []
    for key, cls in sorted(BALT.items(), key=lambda kv: (kv[1], kv[0])):
        rows.append({"vmr_genome_field_normalised": key, "baltimore_class": cls,
                     "rule": "exact match after normalisation (lowercase, whitespace removed)"})
    rows.append({"vmr_genome_field_normalised": "<any other value, or missing>",
                 "baltimore_class": "?",
                 "rule": "unassigned: excluded from the Baltimore probe, retained in the "
                         "feature subset and summarised separately"})
    df = pd.DataFrame(rows)
    df["n_corpus_19429"] = df["baltimore_class"].map(n_corpus).fillna(0).astype(int)
    df["n_feature_subset_1200"] = df["baltimore_class"].map(n_feat).fillna(0).astype(int)
    df["n_baltimore_probe_981"] = df["baltimore_class"].map(n_balt).fillna(0).astype(int)
    df["class_note"] = df["baltimore_class"].map(NOTE)
    # counts are per class, not per key: state it rather than repeat them silently
    df.loc[df["baltimore_class"].duplicated(), ["n_corpus_19429",
                                                "n_feature_subset_1200",
                                                "n_baltimore_probe_981"]] = pd.NA

    df.to_csv(OUT, sep="\t", index=False)
    print(f"[r3] -> {OUT}")
    print(df.to_string(index=False))
    print("\nnormalisation example: norm('ssRNA (+)') ->", repr(norm("ssRNA (+)")))
    print("counts are given once per class, on the first key that maps to it")


if __name__ == "__main__":
    main()
