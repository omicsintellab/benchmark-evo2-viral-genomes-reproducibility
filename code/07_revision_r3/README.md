# Third revision round (R3)

Four controls asked for by an additional reviewer, all computed on CPU from artefacts that
already existed. No GPU work and no new embedding extraction.

| Script | Answers | Writes | Needs |
|---|---|---|---|
| `alpha_length_r3.py` | Does the ridge penalty saturate the grid declared in the manuscript, and is genome length decodable (including on the single-window subset, where pooled-window count is constant)? | `results/json/alpha_length_r3.json` | the cached embedding matrices |
| `contrast_wide_grid_r3.py` | Do the six primary contrasts survive an extended penalty grid, for **both** the embedding and the 6-mer baseline? | `results/json/contrast_wide_grid_r3.json` | the cached embedding matrices + genome FASTA |
| `generation_dispersion_r3.py` | What is the dispersion behind Figure 2, and do the Evo 2 vs Markov differences hold paired within genome? | `results/json/generation_dispersion_r3.json` | `results/generation_*.csv` (shipped) |
| `export_baltimore_mapping.py` | What exactly maps an ICTV VMR genome-type string to a Baltimore class, and what falls through? | `data/baltimore_mapping.tsv` | `data/corpus_manifest.tsv.gz` + `probe_subset_*.tsv` (shipped) |
| `make_supp_table_r3.py` | — (formats the above into Supplementary Tables S11a, S11b and S12) | `results/tables/supplementary_tables_r3.md` | the two JSONs above |

## Positive controls

`alpha_length_r3.py` and `contrast_wide_grid_r3.py` **refuse to report anything** until they
reproduce a published value to six decimals on the published grid — the identity-clustered R²
of gene density (0.775618) for the first, the published deltas of coding fraction
(+0.295107) and gene density (+0.374881) for the second. `make_supp_table_r3.py` asserts that
both JSONs record a passing control before it writes a table. This is deliberate: these
scripts re-derive folds from the same helpers as the published analyses, and a silent drift in
fold construction would otherwise look like a finding.

## Reproducing

`generation_dispersion_r3.py` and `export_baltimore_mapping.py` run from a fresh clone:

```bash
python code/07_revision_r3/generation_dispersion_r3.py
python code/07_revision_r3/export_baltimore_mapping.py
python code/07_revision_r3/make_supp_table_r3.py        # once the two JSONs exist
```

The two ridge analyses read the **cached embedding matrices**, which are not redistributed here
by size (~290 MB for the five 20B layers). Regenerate them with `code/02_embeddings/` on a GPU,
or request them from the authors; then point the scripts at them exactly as the stage-3 analyses
are pointed, since both import their fold machinery from `scale_analysis.py`:

```bash
SCALE_DATA=<dir with manifest/features parquet> SCALE_EMB20B=<dir with the 20B matrices> \
    python code/07_revision_r3/alpha_length_r3.py
```

Their outputs are cached in `results/json/`, so the supplementary tables and every number
quoted in the response letter regenerate without them.

## What changed in the manuscript because of these

- The penalty grid declared in the manuscript is saturated at its upper bound in essentially
  every fold, and in **100%** of folds for the 6-mer baseline, whose optimum lies outside the
  grid (3,162–10,000). On the extended grid the six contrasts keep their direction and stay
  significant after Holm correction under both CV schemes, but the magnitudes fall by 20–25%.
  The published grid remains primary (it was fixed before the data were seen); both are
  reported in Table S11b.
- Log genome length is decodable at R² 0.955 (primary scheme). On the single-window subset it
  is 0.946, so the pooled-window-count route cannot account for the architecture signal.
- Figure 2's panel C changes meaning: on held-out eukaryotic viruses the 4-mer cosine
  similarity of the sampled output is **indistinguishable** from the Markov baseline
  (−0.031 [−0.065, +0.003], Wilcoxon p = 0.31), not below it. For phages the advantage holds
  (+0.108 [0.077, 0.139]).
