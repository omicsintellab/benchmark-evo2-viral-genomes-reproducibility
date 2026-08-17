# Benchmarking Evo 2 on viral genomes — reproducibility repository

Code, curated inputs and cached metrics for:

> **Genomic foundation model embeddings encode higher-order viral genome architecture beyond sequence composition: a benchmark of Evo 2**
> Amgarten D., Schinaid A., de Mello Malta F., Marra A. R., Pinho J. R. R.
>
> Submitted to *Frontiers in Bioinformatics* (Brief Research Report, Genomic Analysis section) on 2026-07-14, to the Research Topic *"Unveiling generative models in microbial genomics: validation, synthetic data, and scalable genome-scale applications"*. A preprint is available on **bioRxiv** (DOI to be added here once registered).

The study benchmarks the **Evo 2 20B base model** (with the 7B model as a scale comparator; neither fine-tuned) on a pre-registered corpus of **19,429 RefSeq viral genomes**, along three axes:

1. **Representation** — linear probes decoding Baltimore class, host domain and viral family from mean-pooled embeddings.
2. **Feature decoding** — ridge probes recovering genome architecture (coding fraction, gene density, gene overlap, …), benchmarked against a 6-mer composition representation and a GC+length control.
3. **Generation** — teacher-forced perplexity and fragment completion on a leakage-safe set of eukaryote-infecting viruses (held out of Evo 2's training corpus) versus a bacteriophage comparator (seen in training).

---

## Reproducing the figures (no GPU required)

The expensive stages (embedding extraction, generation) are separated from the analysis. All cross-validated metrics are cached as small JSON files under [`results/json/`](results/json), so **every figure and table in the paper regenerates from this repository in seconds, on a laptop**:

```bash
conda env create -f environment.yml
conda activate evo2-viral-benchmark

python code/05_figures/make_figure1_combined.py       # Figure 1
python code/05_figures/make_figures_20b.py            # Figure 2 + Table 1
python code/05_figures/make_figure3_combined.py       # Figure 3
python code/05_figures/make_supplementary_tables.py   # Supplementary Tables S1-S4
```

Figures are written to [`figures/`](figures); the metric tables to [`results/tables/`](results/tables). Nothing in the paper's tables is typed by hand — every value is regenerated from the cached JSONs.

### Figure → script → data map

The generated file names keep the identifiers used during analysis and **do not follow the paper's numbering** — `figure3_20b` is Figure 2 of the paper. This table is the authoritative mapping; the figures exactly as submitted are also archived, correctly numbered, under [`figures/final/`](figures/final).

| Paper item | As submitted | Generated as | Produced by | Reads |
|---|---|---|---|---|
| Figure 1 (probes: confusion matrix, accuracy, R², PCA) | `figures/final/figure1.{png,svg}` | `figures/figure1_combined.{png,svg}` | `make_figure1_combined.py` | `fig_artifacts_20b.json`, `scale_metrics.json`, `viral_features_extended_metrics.json` |
| Figure 2 (generation: perplexity, completion) | `figures/final/figure2.{png,svg}` | `figures/figure3_20b.{png,svg}` | `make_figures_20b.py` | `generation_summary_evo2_20b.json` |
| Figure 3 (layer sensitivity + 20B vs 7B scale) | `figures/final/figure3.{png,svg}` | `figures/figure3_combined.{png,svg}` | `make_figure3_combined.py` | `scale_metrics.json`, `pca_control_metrics.json` |
| Table 1 (probe performance ± SD, paired tests) | — | `results/tables/probe_metrics_20b.md` | `make_figures_20b.py` | same as Figure 1 |
| Supplementary Tables S1–S4 | — | `results/tables/supplementary_tables.md` | `make_supplementary_tables.py` | `precision_control_metrics.json`, `scale_metrics.json`, `pca_control_metrics.json`, `data/corpus_manifest.tsv.gz` |

The four supplementary tables report the controls behind the paper's methodological claims: **S1** the FP8-vs-bfloat16 precision control, **S2** random vs sequence-identity cluster-aware cross-validation, **S3** the dimensionality-matched (150-component PCA) comparison, and **S4** the composition of the corpus by Baltimore class and host domain.

---

## Full pipeline

Run in order. Stages 2 and 4 need a GPU with the [Evo 2](https://github.com/ArcInstitute/evo2) runtime installed; stages 1, 3 and 5 do not.

| Stage | Directory | What it does | Hardware |
|---|---|---|---|
| 1. Corpus | [`code/01_corpus/`](code/01_corpus) | Downloads the RefSeq viral release, joins ICTV VMR (MSL41) taxonomy, applies the pre-registered quotas, extracts per-genome features from the GenBank flat files, clusters by sequence identity (MMseqs2 linclust, 95% id / 85% cov). `make_analysis_inputs.py` converts the shipped TSVs into the parquet inputs the later stages read — run it once before stages 2–4. | CPU |
| 2. Embeddings | [`code/02_embeddings/`](code/02_embeddings) | `probe_evo2_viral.py` extracts windowed (32 kb window / 16 kb stride) mean-pooled embeddings and runs the probe battery. `sweep_layers_20b.py` extracts five candidate layers in a single forward pass for the layer-sensitivity analysis. | GPU (H100 for the 20B, FP8; L40S for the 7B, bf16) |
| 3. Analysis | [`code/03_analysis/`](code/03_analysis) | Cluster-aware CV (`cluster_cv.py`), scale comparison (`scale_analysis.py`), dimensionality-matched PCA control (`pca_control.py`), FP8-vs-bf16 precision control (`precision_control.py`), extended features (`viral_features_extended.py`). Emits the JSONs in `results/json/`. | CPU (needs the cached embeddings) |
| 4. Generation | [`code/04_generation/`](code/04_generation) | Teacher-forced perplexity and prompt→gap completion against a 4th-order Markov baseline. | GPU |
| 5. Figures | [`code/05_figures/`](code/05_figures) | Plots and metric tables from the cached JSONs. | CPU |
| 6. Revision R1 | [`code/06_revision_r1/`](code/06_revision_r1) | Two experiments added during peer review: a **block-shuffling control** for long-range genome arrangement, and a **re-run of generation that persists the generated sequences**, with a decoding sweep. Both write to `results/json/`. | GPU (single H100 80 GB; see `README-gpu.md`) |

### Analyses added during peer review

All of these run on CPU from the cached embeddings and emit JSON under `results/json/`, so
every number in the revised manuscript and its supplementary tables regenerates without a GPU.

| Question raised in review | Script | Cached result |
|---|---|---|
| Does the signal survive holding out whole families and genera? | `code/03_analysis/family_cv.py`, `within_family_cv.py` | `family_cv_metrics.json`, `within_family_cv_metrics.json` |
| Do stronger compositional baselines close the gap? (k = 3–6, multi-k, codon, dicodon, six-frame ORF scan) | `code/03_analysis/composition_baselines.py` | `composition_baselines_metrics.json` |
| Is the statistical treatment valid for repeated CV? | `code/03_analysis/final_statistics.py`, `ci_consistency.py` | `final_statistics.json`, `ci_consistency.json` |
| Classification beyond accuracy (macro-F1, balanced accuracy, per-family recall) | `code/03_analysis/classification_metrics.py` | `classification_metrics.json` |
| Does the eukaryote-vs-phage generation gap survive adjustment? | `code/03_analysis/generation_matched.py` | `generation_matched_metrics.json` |
| How often was the gene→CDS fallback used, and does the gene-overlap definition matter? | `code/03_analysis/reextract_annotation.py`, `overlap_sensitivity.py` | `annotation_provenance.json`, `overlap_sensitivity.json` |
| Do the embeddings encode long-range arrangement? | `code/03_analysis/block_shuffle_metrics.py` (from stage 6) | `block_shuffle_metrics_*.json` |
| Supplementary tables | `code/05_figures/make_supp_tables_r1.py`, `make_supp_table_selection.py`, `make_family_cv_table.py`, `make_composition_table.py` | `results/tables/` |

Model checkpoints are **not** redistributed here — obtain `evo2_20b` / `evo2_7b` from the [official Evo 2 release](https://github.com/ArcInstitute/evo2) and pass the path via `--weights-local`.

---

## What is in `data/`

Everything needed to identify the exact genomes analysed, derived entirely from public sources (NCBI RefSeq viral + ICTV VMR). See [`data/README.md`](data/README.md) for the column dictionary and for how to rebuild the FASTA.

| File | Contents |
|---|---|
| `corpus_design.yaml` | The **pre-registration**: quotas, quality cut-offs, dedup and split parameters, fixed before the data were seen. |
| `corpus_manifest.tsv.gz` | The 19,429-genome corpus: accession, family, genus, Baltimore class, host domain, quota group, length. |
| `genome_features.tsv.gz` | Per-genome architectural features (coding fraction, gene density, gene overlap, intergenic statistics, GC, …) — the regression targets. |
| `probe_subset_baltimore.tsv` | The 981 accessions used for the Baltimore probe. |
| `probe_subset_features.tsv` | The 1,200 accessions used for the feature, host and family probes (union with the above: 1,912 genomes). |
| `cl95_cluster.tsv` | MMseqs2 linclust map (95% identity / 85% coverage) used for group-aware cross-validation. |

**Not in this repository, by size:** the genome FASTA (rebuildable from the accession list — see `data/README.md`) and the cached embedding matrices (~30 MB for the 7B; ~290 MB for the five 20B layers). Per the manuscript's Data Availability statement, cached embeddings are **available from the authors on request**.

---

## Notebooks

[`notebooks/`](notebooks) holds the executed exploratory notebooks for the **7B** model (outputs preserved), which preceded the headless scripts used for the 20B results reported in the paper. Paths and bucket names were replaced with placeholders. The authoritative implementations are the scripts under `code/`.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite Evo 2 (Brixi et al., 2025), RefSeq (O'Leary et al., 2016) and the ICTV VMR (Lefkowitz et al., 2018).

## License

MIT — see [`LICENSE`](LICENSE).
