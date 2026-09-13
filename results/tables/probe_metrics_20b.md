# Probe metrics — Evo 2 20B (blocks.18, FP8), mean ± SD over 5×3 repeated CV (Table 1)

Descriptive estimates, cluster-aware CV. Status follows Section 2.8 of the manuscript: *Primary (S6)* = tested with the Nadeau–Bengio corrected resampled t-test and Holm correction on the 1,691 records with an assigned family (`final_statistics.json`); *Negative control* = pre-declared, outside the correction; *Exploratory* = no confirmatory test.

1. Repeated **random** (not cluster-aware) CV, from `viral_features_extended_metrics.json`; the two schemes differ by at most 0.012 on every target evaluated under both (Supplementary Table S4).

| Probe | Target | Evo 2 20B (blocks.18) | 6-mer composition | GC + length | Status |
|---|---|---|---|---|---|
| Class. (acc) | Baltimore | 0.961 ± 0.011 | 0.808 ± 0.023 | 0.468 ± 0.026 | Exploratory |
| Class. (acc) | Host | 0.995 ± 0.005 | 0.934 ± 0.018 | 0.816 ± 0.019 | Exploratory |
| Class. (acc) | Family | 0.907 ± 0.032 | 0.891 ± 0.025 | 0.354 ± 0.049 | Exploratory |
| Regr. (R²) | coding_fraction | 0.606 ± 0.064 | 0.100 ± 0.078 | 0.024 ± 0.026 | Primary (S6) |
| Regr. (R²) | gene_density | 0.766 ± 0.033 | 0.380 ± 0.062 | 0.067 ± 0.030 | Primary (S6) |
| Regr. (R²) | noncoding_bp | 0.752 ± 0.036 | 0.254 ± 0.070 | 0.560 ± 0.043 | Primary (S6) |
| Regr. (R²) | n_genes | 0.908 ± 0.029 | 0.427 ± 0.067 | 0.755 ± 0.055 | Primary (S6) |
| Regr. (R²) | mean_intergenic_len | 0.556 ± 0.050 | 0.091 ± 0.069 | 0.253 ± 0.022 | Primary (S6) |
| Regr. (R²) | cpg_oe ¹ | 0.957 ± 0.009 | 0.961 ± 0.005 | 0.238 ± 0.040 | Negative control |
| Regr. (R²) | upa_oe ¹ | 0.889 ± 0.015 | 0.936 ± 0.008 | 0.091 ± 0.036 | Negative control |
| Regr. (R²) | overlap_bp ¹ | 0.643 ± 0.040 | 0.265 ± 0.053 | 0.348 ± 0.046 | Primary (S6) |
