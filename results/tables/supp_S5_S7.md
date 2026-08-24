## Supplementary Table S5. Primary contrasts under both cross-validation schemes

Evo 2 20B `blocks.18` versus the 6-mer baseline on the six pre-declared architecture targets. ΔR² is the difference in cross-validated R²; the test is a Nadeau–Bengio corrected resampled t-test and the interval is obtained by inverting the same statistic. p-values are Holm-corrected within the set of six.

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

## Supplementary Table S6. Evo 2 versus the best baseline of each class

For each target and scheme, the strongest baseline within each class was selected and contrasted with the embedding. Compositional: k-mer (k = 3–6), multi-k, codon, dicodon, GC-and-length. Annotation-derived: six-frame ORF scan and the combined representation containing it. A positive ΔR² favours Evo 2.

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

## Supplementary Table S7. Classification beyond accuracy

Under the pre-registered identity-clustered scheme. Balanced accuracy and macro-F1 are reported because the classes are unbalanced (R2 #15).

| Target | n | Classes | Representation | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| baltimore | 981 | 7 | `evo2_20b_blocks18` | 0.962 | 0.963 | 0.963 |
| baltimore | 981 | 7 | `6mer` | 0.814 | 0.822 | 0.819 |
| host | 1080 | 3 | `evo2_20b_blocks18` | 0.993 | 0.992 | 0.986 |
| host | 1080 | 3 | `6mer` | 0.929 | 0.917 | 0.880 |
| family | 349 | 8 | `evo2_20b_blocks18` | 0.906 | 0.892 | 0.893 |
| family | 349 | 8 | `6mer` | 0.899 | 0.883 | 0.885 |

