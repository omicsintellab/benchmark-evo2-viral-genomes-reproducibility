# Data

All files here derive from **public sources**: the NCBI RefSeq viral release (May 2026) and the ICTV Virus Metadata Resource (MSL41). No sequence data are redistributed — only accessions, taxonomy and annotation-derived statistics.

## `corpus_design.yaml` — the versioned design document

Quotas (Baltimore class × host domain), quality cut-offs, sanitisation filters, dedup parameters (MMseqs2: 95% identity, 85% coverage) and split seeds, fixed before the data were inspected. It is a versioned file in this repository, not a registration deposited with a third-party registry, and the manuscript describes it in those terms.

**Scope for this benchmark.** The file was written for the project's fine-tuning corpus, and the division is visible in its own structure rather than argued after the fact: the identity/coverage parameters and the cluster-aware split requirement that this benchmark uses sit in the block **common to both corpora**, whereas `min_length: 1000` and the completeness filter sit **inside the `v1_refseq:` block** and the ambiguous-base cut-off sits in the sanitisation stage of that pipeline. The benchmark therefore uses its **quota groups** and its **identity-clustering parameters** (the `quota_group` column and `cl95_cluster.tsv`), and follows its requirement of cluster-aware splitting. It does **not** apply the fine-tuning cut-offs: `min_length: 1000` and `sanitize.max_N_fraction` are enforced only in `code/01_corpus/build_v1_selection.py`, so the corpus manifest keeps 573 records shorter than 1,000 bp, 63 of them in the probe subsets (shortest 266 bp). The `split` block (90/5/5 train/val/test) is the fine-tuning split; the probes use 5-fold × 3-repeat cross-validation, which was not pre-registered. The `v2` changelog entry (2026-06-29) changes the fine-tuning balancing policy only. `meta.version` is bumped on any change, with the rationale recorded in the changelog block and in git history. Consumed by `code/01_corpus/compose_v1_groups.py`.

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
| `overlap_bp` | Summed length of all CDS loci minus the length of their union: a position covered by *k* CDS counts *k* − 1 times, strands not separated (a hallmark of genome compression in small viruses). Log-transformed for probing. This is the definition used in every main analysis; two alternatives (positions covered by ≥2 CDS; same-strand only) are compared in `overlap_sensitivity.json` / Supplementary Table S8. |
| `has_introns`, `strand_balance` | Presence of spliced CDSs; balance of CDSs between strands. |
| `family`, `genus`, `host` | Denormalised taxonomy, for convenience. |

The CpG and UpA observed/expected dinucleotide ratios used as low-order composition controls are **not** in this table: they are computed from the genome sequence at analysis time by `code/03_analysis/viral_features_extended.py` (genomes with fewer than 100 resolved ACGT bases are excluded).

## `probe_subset_baltimore.tsv` / `probe_subset_features.tsv`

The exact genomes drawn into each probe, so that results can be reproduced on the same samples rather than on a fresh draw:

- **Baltimore probe** — n = 981 across seven classes, with **unequal** per-class counts: the 150-per-class quota is met in classes I–V, but only 94 records of class VI and 137 of class VII exist in the corpus after the Baltimore mapping, so those classes are taken whole.
- **Feature / host / family probes** — 120 genomes per quota group, n = 1,200 (family restricted to families with ≥ 25 members: n = 349).
- Union of the two subsets: **1,912 genomes** — the set embedded for the layer sweep and the scale comparison.

## `baltimore_mapping.tsv` — how a VMR genome-type string becomes a Baltimore class

Exported by `code/07_revision_r3/export_baltimore_mapping.py` from the dictionary used to build the corpus (`code/01_corpus/compose_v1_groups.py`, `BALT`), so the table cannot drift from the code.

The rule is an **exact match** on the ICTV VMR `Genome` field after normalisation (lowercased, whitespace removed). Strings that match no key are left **unassigned**:

| Column | Description |
|---|---|
| `vmr_genome_field_normalised` | The normalised VMR string, or `<any other value, or missing>` for the residual row. |
| `baltimore_class` | `I`–`VII`, or `?` for unassigned. |
| `rule` | The matching rule applied. |
| `n_corpus_19429`, `n_feature_subset_1200`, `n_baltimore_probe_981` | Counts the rule yields in each population (given once per class, on its first row). |
| `class_note` | Which VMR strings the class accepts. |

**Unassigned records** — 3,238 of the 19,429 corpus records (16.7%) and 240 of the 1,200 in the feature subset (20.0%) — are **excluded from the Baltimore probe** and **retained** in the feature subset, where they are summarised separately. This is where ambisense genomes (*Arenaviridae*, segmented families of *Bunyavirales*) and unsigned-ssRNA records fall: the VMR records them with strings the dictionary does not key, and they are not forced into class IV or V.

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
