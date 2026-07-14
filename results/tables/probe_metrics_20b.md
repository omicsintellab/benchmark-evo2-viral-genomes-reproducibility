# Probe metrics — Evo 2 20B (blocks.18, FP8), cluster-aware CV (5×3 repeats, mean ± SD)

Significance: paired t-test across the 15 fold values; *** p<0.001, ** p<0.01, * p<0.05, ns.

1. CpG/UpA/overlap probes were run with only mean/SD cached (no per-fold significance test).

| Probe | Target | Evo 2 20B (blocks.18) | 6-mer composition | GC + length | 20B vs k-mer | 20B vs GC+len |
|---|---|---|---|---|---|---|
| Class. (acc) | Baltimore | 0.961 ± 0.011 | 0.808 ± 0.023 | 0.468 ± 0.026 | *** | *** |
| Class. (acc) | Host | 0.995 ± 0.005 | 0.934 ± 0.018 | 0.816 ± 0.019 | *** | *** |
| Class. (acc) | Family | 0.907 ± 0.032 | 0.891 ± 0.025 | 0.354 ± 0.049 | ns | *** |
| Regr. (R²) | coding_fraction | 0.606 ± 0.064 | 0.100 ± 0.078 | 0.024 ± 0.026 | *** | *** |
| Regr. (R²) | gene_density | 0.766 ± 0.033 | 0.380 ± 0.062 | 0.067 ± 0.030 | *** | *** |
| Regr. (R²) | noncoding_bp | 0.752 ± 0.036 | 0.254 ± 0.070 | 0.560 ± 0.043 | *** | *** |
| Regr. (R²) | n_genes | 0.908 ± 0.029 | 0.427 ± 0.067 | 0.755 ± 0.055 | *** | *** |
| Regr. (R²) | mean_intergenic_len | 0.556 ± 0.050 | 0.091 ± 0.069 | 0.253 ± 0.022 | *** | *** |
| Regr. (R²) | cpg_oe | 0.957 ± 0.009 | 0.961 ± 0.005 | 0.238 ± 0.040 | n/a¹ | n/a¹ |
| Regr. (R²) | upa_oe | 0.889 ± 0.015 | 0.936 ± 0.008 | 0.091 ± 0.036 | n/a¹ | n/a¹ |
| Regr. (R²) | overlap_bp | 0.643 ± 0.040 | 0.265 ± 0.053 | 0.348 ± 0.046 | n/a¹ | n/a¹ |
