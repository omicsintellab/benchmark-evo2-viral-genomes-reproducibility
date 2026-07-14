# Data

All files here derive from **public sources**: the NCBI RefSeq viral release (May 2026) and the ICTV Virus Metadata Resource (MSL41). No sequence data are redistributed — only accessions, taxonomy and annotation-derived statistics.

## `corpus_design.yaml` — the pre-registration

Quotas (Baltimore class × host domain), quality cut-offs, sanitisation filters, dedup parameters (MMseqs2: 95% identity, 85% coverage) and split seeds, all fixed **before** the data were inspected. `meta.version` is bumped on any change, with the rationale recorded in the changelog block and in git history. Consumed by `code/01_corpus/compose_v1_groups.py`.

## `corpus_manifest.tsv.gz` — the 19,429-genome corpus

One row per RefSeq record (segmented viruses contribute one row per segment).

| Column | Description |
|---|---|
| `accession` | RefSeq accession (e.g. `NC_029549.1`) — the join key for every other file. |
| `family`, `genus` | ICTV taxonomy, joined from the VMR (MSL41). |
| `baltimore` | Baltimore replication class, `I`–`VII`. |
| `host` | Host domain: `eukaryote`, `bacterium`, `archaeon`. |
| `quota_group` | Pre-registered stratum (Baltimore × host), e.g. `ssRNA_neg_euk`. |
| `is_recent` | Family absent from the baseline VMR (MSL38) and present in MSL41, i.e. created in 2023+ — eukaryotic hosts only. |
| `length` | Genome (or segment) length in bp. |

## `genome_features.tsv.gz` — annotation-derived features

Computed per record directly from the GenBank flat files with Biopython (`code/01_corpus/extract_genome_features.py`). These are the **regression targets** of the feature probes.

| Column | Description |
|---|---|
| `accession`, `organism` | Identity. |
| `genome_length`, `GC` | Length in bp; GC fraction over the full genome. |
| `n_CDS`, `n_genes`, `n_tRNA`, `n_rRNA`, `n_ncRNA` | Annotated feature counts. |
| `coding_bp` | Length of the **union** of all CDS exon intervals (join-aware: multi-exon or spliced CDSs counted once). |
| `coding_fraction` | `coding_bp / genome_length`, bounded in [0, 1]. |
| `noncoding_bp` | `genome_length − coding_bp`. |
| `gene_density` | Annotated gene features per kb (falls back to the CDS count when a record has no explicit gene features). |
| `mean_CDS_len` | Mean CDS length. |
| `n_intergenic`, `mean_intergenic_len`, `median_intergenic_len`, `max_intergenic_len` | Gaps between consecutive, non-overlapping CDS loci. |
| `overlap_bp` | Summed length of all CDS loci minus the length of their union — base pairs covered by more than one CDS (a hallmark of genome compression in small viruses). Log-transformed for probing. |
| `has_introns`, `strand_balance` | Presence of spliced CDSs; balance of CDSs between strands. |
| `family`, `genus`, `host` | Denormalised taxonomy, for convenience. |

The CpG and UpA observed/expected dinucleotide ratios used as low-order composition controls are **not** in this table: they are computed from the genome sequence at analysis time by `code/03_analysis/viral_features_extended.py` (genomes with fewer than 100 resolved ACGT bases are excluded).

## `probe_subset_baltimore.tsv` / `probe_subset_features.tsv`

The exact genomes drawn into each probe, so that results can be reproduced on the same samples rather than on a fresh draw:

- **Baltimore probe** — 150 genomes per class, n = 981 across seven classes.
- **Feature / host / family probes** — 120 genomes per quota group, n = 1,200 (family restricted to families with ≥ 25 members: n = 349).
- Union of the two subsets: **1,912 genomes** — the set embedded for the layer sweep and the scale comparison.

## `cl95_cluster.tsv`

MMseqs2 `linclust` output (95% identity, 85% coverage, greedy set-cover) over the 1,912 probed genomes: `cluster_representative<TAB>member`. Used as the grouping variable for group-aware cross-validation, so that no cluster spans training and held-out folds. Only 16 of 1,912 genomes merged, reflecting RefSeq's one-reference-per-species design.

## Rebuilding what is not shipped here

**Genome FASTA** (needed by the 6-mer baselines, the CpG/UpA controls and the embedding stage) — rebuild from the accession list:

```bash
bash code/01_corpus/download_refseq_viral.sh     # fetches the RefSeq viral release (.gbff.gz)
python code/01_corpus/extract_genome_features.py # regenerates genome_features + FASTA
```

Note that RefSeq is a moving target: to reproduce the exact corpus, use the **May 2026** release and the accessions in `corpus_manifest.tsv.gz` rather than whatever release is current.

**Cached embeddings** (~30 MB for the 7B; ~290 MB for the five 20B layers, `float32`, 8,192-d) — regenerate with `code/02_embeddings/` on a GPU, or request them from the authors (see the manuscript's Data Availability statement).
