**Supplementary Material**

*Decoding taxonomy and genome-level architecture from Evo 2 embeddings of viral genomes: a linear-probe benchmark against compositional baselines*

Tables are numbered in order of first citation in the manuscript.

**1 Supplementary Tables**

**Supplementary Table S1.** Composition of the viral genome corpus (n = 19,429 RefSeq records, including individual segments of segmented viruses), cross-tabulated by Baltimore replication class and host domain as assigned through the ICTV Virus Metadata Resource (MSL41). Records whose family- or genus-level taxonomy could not be mapped to a Baltimore class or to a host domain in the VMR are reported as unassigned/unknown. Probe subsets were drawn as balanced samples from these quota groups, so that the strong imbalance of the full corpus — for example, 6,311 class I records versus 94 class VI — does not propagate into the probes.

| Baltimore class | Eukaryote | Bacteria | Archaea | Unknown | Total |
|---|---|---|---|---|---|
| I (dsDNA) | 1,003 | 5,179 | 129 | 0 | 6,311 |
| II (ssDNA) | 2,283 | 132 | 17 | 0 | 2,432 |
| III (dsRNA) | 1,627 | 36 | 0 | 0 | 1,663 |
| IV (ssRNA+) | 2,715 | 834 | 0 | 0 | 3,549 |
| V (ssRNA−) | 2,005 | 0 | 0 | 0 | 2,005 |
| VI (ssRNA-RT) | 94 | 0 | 0 | 0 | 94 |
| VII (dsDNA-RT) | 137 | 0 | 0 | 0 | 137 |
| Unassigned | 712 | 0 | 0 | 2,526 | 3,238 |
| **Total** | **10,576** | **6,181** | **146** | **2,526** | **19,429** |

**Supplementary Table S2.** Composition of every analysed population. Counts are derived from the released data files, not transcribed. It resolves the apparent inconsistency between 981, 1,080 and 1,912.

| Stage | n | Excluded at this step | Composition and use |
|---|---|---|---|
| Pre-registered RefSeq viral corpus | 19,429 | — | quota sampling by Baltimore class and host domain |
| Baltimore probe subset | 981 | — | I 150, II 150, III 150, IV 150, V 150, VI 94, VII 137; classes VI and VII have fewer than 150 records meeting the quota |
| Feature probe subset | 1,200 | — | regression targets for genome architecture |
| Host-domain classification | 1,080 | −120 unknown host domain | 840 eukaryote, 120 bacteria, 120 archaea |
| Union of both subsets | 1,912 | — | layer sweep, scale and precision controls; intersection of the two subsets = 269 |
| Family-grouped analyses | 1,691 | −221 without an assigned family | GroupKFold by family, leave-one-family-out, within-family CV |

**Segmented viruses.** Features were extracted one row per record, so a segmented virus contributes one row per segment: within the union, 109 organisms contribute more than one record, totalling 268 records (14.0%). In the grouped analyses family subsumes organism, so all records of one organism fall in the same fold; the analysis script aborts if that invariant is violated.

**Records without a family assignment** (221) are excluded from the grouped analyses and reported as a sensitivity analysis treating each as its own group.

**Supplementary Table S3.** Precision control on the NVIDIA H100. Cross-validated probe performance (repeated cluster-aware cross-validation, mean ± SD) for the Evo 2 20B embedding extracted with native FP8 input projections versus the same layers re-extracted with FP8 disabled (forced bfloat16), on identical genomes and folds. Classification reports accuracy; regression reports R². Forcing bfloat16 degrades every target, most severely the fine-grained architectural features, indicating that the 20B checkpoint is calibrated for FP8 inference rather than precision-agnostic.

| Probe target | blocks.15 FP8 | blocks.15 bfloat16 | blocks.18 FP8 | blocks.18 bfloat16 |
|---|---|---|---|---|
| Baltimore class (acc) | 0.962 ± 0.007 | 0.835 ± 0.020 | 0.961 ± 0.011 | 0.830 ± 0.018 |
| Host domain (acc) | 0.992 ± 0.004 | 0.944 ± 0.012 | 0.995 ± 0.005 | 0.940 ± 0.015 |
| Family (acc) | 0.897 ± 0.023 | 0.835 ± 0.039 | 0.907 ± 0.032 | 0.854 ± 0.034 |
| coding_fraction (R²) | 0.606 ± 0.069 | 0.196 ± 0.072 | 0.606 ± 0.064 | 0.240 ± 0.075 |
| gene_density (R²) | 0.793 ± 0.038 | 0.417 ± 0.100 | 0.766 ± 0.033 | 0.453 ± 0.079 |
| noncoding_bp, log (R²) | 0.757 ± 0.044 | 0.594 ± 0.049 | 0.752 ± 0.036 | 0.620 ± 0.042 |
| n_genes, log (R²) | 0.903 ± 0.034 | 0.818 ± 0.046 | 0.908 ± 0.029 | 0.838 ± 0.041 |
| mean_intergenic_len, log (R²) | 0.556 ± 0.052 | 0.273 ± 0.082 | 0.556 ± 0.050 | 0.337 ± 0.046 |

**Supplementary Table S4.** Effect of sequence-identity-aware cross-validation. Probe performance under random repeated cross-validation versus group-aware cross-validation in which MMseqs2 linclust clusters (95% identity, 85% coverage) are used as groups, so that no cluster spans training and held-out folds. Δ is the change induced by the group-aware scheme. Only 16 of the 1,912 probed genomes merged into clusters, and no metric changes by more than 0.012 in either model. Identity clustering addresses near-duplicate leakage only; phylogenetic dependence is addressed by the family-grouped analyses of Supplementary Tables S6 and S7.

| Probe target | 20B random CV | 20B cluster-aware CV | Δ | 7B random CV | 7B cluster-aware CV | Δ |
|---|---|---|---|---|---|---|
| Baltimore class (acc) | 0.957 ± 0.016 | 0.961 ± 0.011 | +0.004 | 0.890 ± 0.015 | 0.883 ± 0.025 | -0.007 |
| Host domain (acc) | 0.993 ± 0.006 | 0.995 ± 0.005 | +0.002 | 0.973 ± 0.007 | 0.971 ± 0.009 | -0.002 |
| Family (acc) | 0.913 ± 0.017 | 0.907 ± 0.032 | -0.006 | 0.860 ± 0.035 | 0.857 ± 0.030 | -0.003 |
| coding_fraction (R²) | 0.604 ± 0.033 | 0.606 ± 0.064 | +0.002 | 0.625 ± 0.056 | 0.618 ± 0.066 | -0.007 |
| gene_density (R²) | 0.771 ± 0.036 | 0.766 ± 0.033 | -0.005 | 0.699 ± 0.047 | 0.691 ± 0.050 | -0.008 |
| noncoding_bp, log (R²) | 0.745 ± 0.039 | 0.752 ± 0.036 | +0.007 | 0.692 ± 0.043 | 0.692 ± 0.029 | -0.000 |
| n_genes, log (R²) | 0.898 ± 0.028 | 0.908 ± 0.029 | +0.010 | 0.823 ± 0.029 | 0.825 ± 0.039 | +0.002 |
| mean_intergenic_len, log (R²) | 0.552 ± 0.034 | 0.556 ± 0.050 | +0.004 | 0.355 ± 0.079 | 0.367 ± 0.059 | +0.012 |

**Supplementary Table S5.** Classification beyond accuracy, under the pre-registered identity-clustered scheme. Balanced accuracy and macro-F1 are reported because the classes are unbalanced.

| Target | n | Classes | Representation | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| baltimore | 981 | 7 | `evo2_20b_blocks18` | 0.962 | 0.963 | 0.963 |
| baltimore | 981 | 7 | `6mer` | 0.814 | 0.822 | 0.819 |
| host | 1080 | 3 | `evo2_20b_blocks18` | 0.993 | 0.992 | 0.986 |
| host | 1080 | 3 | `6mer` | 0.929 | 0.917 | 0.880 |
| family | 349 | 8 | `evo2_20b_blocks18` | 0.906 | 0.892 | 0.893 |
| family | 349 | 8 | `6mer` | 0.899 | 0.883 | 0.885 |

**Per-family recall** (Evo 2 20B `blocks.18`, same folds):

| Family | n | Recall |
|---|---|---|
| *Geminiviridae* | 108 | 1.000 |
| *Papillomaviridae* | 78 | 1.000 |
| *Retroviridae* | 138 | 0.978 |
| *Caulimoviridae* | 168 | 0.976 |
| *Phenuiviridae* | 249 | 0.960 |
| *Peribunyaviridae* | 81 | 0.815 |
| *Sedoreoviridae* | 123 | 0.732 |
| *Spinareoviridae* | 102 | 0.676 |

**Supplementary Table S6.** Primary confirmatory contrasts under both cross-validation schemes. Evo 2 20B `blocks.18` versus the 6-mer baseline on the six pre-declared architecture targets. ΔR² is the difference in cross-validated R²; the test is a Nadeau–Bengio corrected resampled t-test and the interval is obtained by inverting the same statistic. p-values are Holm-corrected within the set of six.

| Scheme | Target | ΔR² | 95% CI | Cohen's d | p (Holm) |
|---|---|---|---|---|---|
| Identity-clustered (pre-registered) | Coding fraction | +0.295 | [+0.195; +0.395] | 3.6 | <1e-4 |
| Identity-clustered (pre-registered) | Gene density | +0.375 | [+0.309; +0.441] | 6.9 | <1e-4 |
| Identity-clustered (pre-registered) | Non-coding bp | +0.512 | [+0.419; +0.605] | 6.7 | <1e-4 |
| Identity-clustered (pre-registered) | Gene count | +0.521 | [+0.446; +0.596] | 8.4 | <1e-4 |
| Identity-clustered (pre-registered) | Mean intergenic length | +0.432 | [+0.357; +0.508] | 6.9 | <1e-4 |
| Identity-clustered (pre-registered) | Gene overlap | +0.406 | [+0.367; +0.446] | 12.4 | <1e-4 |
| Family-grouped | Coding fraction | +0.319 | [+0.110; +0.529] | 1.8 | 0.017 |
| Family-grouped | Gene density | +0.438 | [+0.265; +0.611] | 3.0 | 0.00054 |
| Family-grouped | Non-coding bp | +0.708 | [+0.350; +1.067] | 2.4 | 0.0042 |
| Family-grouped | Gene count | +0.676 | [+0.210; +1.141] | 1.8 | 0.017 |
| Family-grouped | Mean intergenic length | +0.567 | [+0.241; +0.893] | 2.1 | 0.0089 |
| Family-grouped | Gene overlap | +0.387 | [+0.080; +0.694] | 1.5 | 0.017 |

**Supplementary Table S7.** Evo 2 versus the strongest baseline of each class. For each target and scheme, the best baseline within each class was selected and contrasted with the embedding. Compositional: k-mer (k = 3–6), multi-k, codon, dicodon, GC-and-length. Annotation-derived: the six-frame ORF scan and the combined representation containing it. A positive ΔR² favours Evo 2.

| Scheme | Target | Baseline class | Best baseline | Baseline R² | ΔR² | 95% CI | p |
|---|---|---|---|---|---|---|---|
| Identity-clustered (pre-registered) | Coding fraction | Compositional | `kmer5` | 0.153 | +0.223 | [+0.151; +0.295] | <1e-4 |
| Identity-clustered (pre-registered) | Coding fraction | Annotation-derived | `orf` | 0.223 | +0.153 | [+0.077; +0.230] | 0.00074 |
| Identity-clustered (pre-registered) | Gene density | Compositional | `dicodon` | 0.433 | +0.342 | [+0.259; +0.425] | <1e-4 |
| Identity-clustered (pre-registered) | Gene density | Annotation-derived | `orf` | 0.588 | +0.188 | [+0.140; +0.235] | <1e-4 |
| Identity-clustered (pre-registered) | Non-coding bp | Compositional | `gc_len` | 0.549 | +0.217 | [+0.148; +0.287] | <1e-4 |
| Identity-clustered (pre-registered) | Non-coding bp | Annotation-derived | `orf` | 0.675 | +0.091 | [+0.036; +0.146] | 0.0032 |
| Identity-clustered (pre-registered) | Gene count | Compositional | `gc_len` | 0.739 | +0.157 | [+0.118; +0.195] | <1e-4 |
| Identity-clustered (pre-registered) | Gene count | Annotation-derived | `orf` | 0.900 | -0.005 | [-0.030; +0.020] | 0.66 |
| Identity-clustered (pre-registered) | Mean intergenic length | Compositional | `gc_len` | 0.239 | +0.315 | [+0.266; +0.364] | <1e-4 |
| Identity-clustered (pre-registered) | Mean intergenic length | Annotation-derived | `orf` | 0.335 | +0.219 | [+0.186; +0.252] | <1e-4 |
| Identity-clustered (pre-registered) | Gene overlap | Compositional | `gc_len` | 0.285 | +0.338 | [+0.290; +0.385] | <1e-4 |
| Identity-clustered (pre-registered) | Gene overlap | Annotation-derived | `orf` | 0.387 | +0.236 | [+0.188; +0.283] | <1e-4 |
| Family-grouped | Coding fraction | Compositional | `kmer4` | 0.007 | +0.199 | [+0.016; +0.383] | 0.035 |
| Family-grouped | Coding fraction | Annotation-derived | `orf` | 0.182 | +0.025 | [-0.085; +0.134] | 0.64 |
| Family-grouped | Gene density | Compositional | `kmer5` | 0.096 | +0.436 | [+0.276; +0.597] | <1e-4 |
| Family-grouped | Gene density | Annotation-derived | `orf` | 0.520 | +0.012 | [-0.110; +0.134] | 0.84 |
| Family-grouped | Non-coding bp | Compositional | `gc_len` | 0.539 | +0.060 | [-0.015; +0.135] | 0.11 |
| Family-grouped | Non-coding bp | Annotation-derived | `orf` | 0.643 | -0.044 | [-0.132; +0.044] | 0.3 |
| Family-grouped | Gene count | Compositional | `gc_len` | 0.722 | +0.061 | [-0.040; +0.162] | 0.22 |
| Family-grouped | Gene count | Annotation-derived | `orf` | 0.891 | -0.109 | [-0.196; -0.021] | 0.018 |
| Family-grouped | Mean intergenic length | Compositional | `gc_len` | 0.206 | +0.126 | [+0.017; +0.235] | 0.027 |
| Family-grouped | Mean intergenic length | Annotation-derived | `orf` | 0.272 | +0.060 | [-0.076; +0.195] | 0.36 |
| Family-grouped | Gene overlap | Compositional | `gc_len` | 0.246 | +0.092 | [-0.063; +0.246] | 0.22 |
| Family-grouped | Gene overlap | Annotation-derived | `orf` | 0.318 | +0.020 | [-0.095; +0.134] | 0.72 |

**Summary.** Under the pre-registered scheme Evo 2 leads the compositional class on 6 of 6 targets and the annotation-derived class on 5 of 6. Under family-grouped cross-validation it leads the compositional class on 3 of 6 and the annotation-derived class on 0 of 6, and is significantly behind the ORF baseline on gene count.

**Supplementary Table S8.** Sensitivity of the gene-overlap contrast to the definition of the feature (n = 1,691). The published definition counts a position once per overlapping CDS pair and does not separate strands; *positions* counts genome positions covered by two or more CDS features; *same-strand* restricts to overlap between features on the same strand. Evo 2 20B `blocks.18` versus the 6-mer baseline. The conclusion holds under all three definitions in both schemes.

| Definition | Scheme | Evo 2 R² | 6-mer R² | ΔR² | p |
|---|---|---|---|---|---|
| Published | Identity-clustered (pre-registered) | 0.623 | 0.216 | +0.406 | <1e-4 |
| Published | Family-grouped | 0.338 | -0.049 | +0.387 | 0.017 |
| Positions covered | Identity-clustered (pre-registered) | 0.622 | 0.216 | +0.406 | <1e-4 |
| Positions covered | Family-grouped | 0.340 | -0.048 | +0.388 | 0.017 |
| Same strand | Identity-clustered (pre-registered) | 0.616 | 0.226 | +0.390 | <1e-4 |
| Same strand | Family-grouped | 0.300 | -0.050 | +0.350 | 0.034 |

**Supplementary Table S9.** Dimensionality-matched comparison. Because the 20B embedding is higher-dimensional than the 7B (8,192 versus 4,096 components), which could by itself inflate linear-probe performance, every probe was re-run with both embeddings projected onto a common 150-component PCA subspace, fitted inside each cross-validation training fold to prevent leakage. The number of components follows from min(n_train, n_features), with approximately 279 training samples per fold for the family target. All values are cluster-aware cross-validation, mean ± SD.

| Probe target | 20B full (8,192-d) | 20B PCA-150 | 7B full (4,096-d) | 7B PCA-150 |
|---|---|---|---|---|
| Baltimore class (acc) | 0.961 ± 0.011 | 0.958 ± 0.009 | 0.883 ± 0.025 | 0.883 ± 0.023 |
| Host domain (acc) | 0.995 ± 0.005 | 0.991 ± 0.005 | 0.971 ± 0.009 | 0.971 ± 0.010 |
| Family (acc) | 0.907 ± 0.032 | 0.913 ± 0.031 | 0.857 ± 0.030 | 0.856 ± 0.032 |
| coding_fraction (R²) | 0.606 ± 0.064 | 0.597 ± 0.063 | 0.618 ± 0.066 | 0.617 ± 0.065 |
| gene_density (R²) | 0.766 ± 0.033 | 0.733 ± 0.032 | 0.691 ± 0.050 | 0.664 ± 0.045 |
| noncoding_bp, log (R²) | 0.752 ± 0.036 | 0.722 ± 0.036 | 0.692 ± 0.029 | 0.665 ± 0.031 |
| n_genes, log (R²) | 0.908 ± 0.029 | 0.875 ± 0.028 | 0.825 ± 0.039 | 0.803 ± 0.043 |
| mean_intergenic_len, log (R²) | 0.556 ± 0.050 | 0.510 ± 0.051 | 0.367 ± 0.059 | 0.333 ± 0.048 |

**Supplementary Table S10.** Family-level generalisation. Leave-one-family-out is restricted to families with at least 30 genomes and summarised by R² over pooled out-of-fold predictions. Within-family cross-validation is run inside each such family and summarised by the median across families; it separates family-level from within-family signal, which leave-one-family-out conflates. SD ratio is the median within-family standard deviation of the target divided by its global standard deviation — a low ratio means restriction of range, which bounds the attainable R² regardless of representation.

| Target | LOFO pooled R² (Evo 2) | Within-family median R² (Evo 2) | Within-family median R² (6-mer) | Families | Median SD ratio |
|---|---|---|---|---|---|
| Coding fraction | 0.104 | 0.124 | 0.076 | 10 | 0.85 |
| Gene density | 0.303 | 0.426 | 0.003 | 10 | 0.70 |
| Non-coding bp | 0.353 | 0.213 | 0.053 | 10 | 0.56 |
| Gene count | 0.023 | -0.130 | 0.047 | 10 | 0.13 |
| Mean intergenic length | 0.139 | 0.058 | 0.233 | 7 | 0.89 |
| Gene overlap | 0.313 | -0.242 | -0.073 | 9 | 0.52 |

