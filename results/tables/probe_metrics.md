---
title: "Probe metrics — repeated 5-fold CV (mean ± SD)"
stage: "09"
type: result
tags: ["stage/09", "type/result", "domain/results"]
aliases: []
status: active
---
# Probe metrics — repeated 5-fold CV (mean ± SD)

Significance: paired t-test across folds; *** p<0.001, ** p<0.01, * p<0.05, ns.

| Probe | Target | n | Evo 2 embedding | 6-mer composition | GC + length | Evo 2 vs k-mer | Evo 2 vs GC+len |
|---|---|---|---|---|---|---|---|
| Class. (acc) | Baltimore | 981 | 0.890 ± 0.015 | 0.810 ± 0.028 | 0.460 ± 0.042 | *** | *** |
| Class. (acc) | Host domain | 1080 | 0.973 ± 0.008 | 0.934 ± 0.013 | 0.815 ± 0.034 | *** | *** |
| Class. (acc) | Family | 349 | 0.856 ± 0.033 | 0.909 ± 0.019 | 0.358 ± 0.055 | *** | *** |
| Regr. (R²) | coding_fraction | 1200 | 0.623 ± 0.056 | 0.140 ± 0.077 | 0.029 ± 0.019 | *** | *** |
| Regr. (R²) | gene_density | 1200 | 0.699 ± 0.047 | 0.368 ± 0.082 | 0.064 ± 0.037 | *** | *** |
| Regr. (R²) | noncoding_bp | 1200 | 0.696 ± 0.043 | 0.237 ± 0.120 | 0.556 ± 0.048 | *** | *** |
| Regr. (R²) | n_genes | 1200 | 0.825 ± 0.031 | 0.432 ± 0.070 | 0.755 ± 0.041 | *** | *** |
| Regr. (R²) | mean_intergenic_len | 1200 | 0.364 ± 0.072 | 0.103 ± 0.059 | 0.252 ± 0.042 | *** | *** |

## Cluster-aware CV (anti-leakage)

Genomas dos probes clusterizados por identidade (MMseqs2 linclust, 95% id / 80% cov): **1896 clusters de 1912** (só 16 quase-duplicatas). CV agrupada (StratifiedGroupKFold) vs aleatória — embedding:

| Tarefa | CV aleatória | CV cluster-aware (95%) | Δ |
|---|--:|--:|--:|
| Baltimore (acc) | 0.890 ± 0.015 | 0.885 ± 0.024 | −0.005 |
| Host (acc) | 0.973 ± 0.008 | 0.972 ± 0.009 | −0.001 |
| Family (acc) | 0.856 ± 0.033 | 0.857 ± 0.026 | +0.001 |
| coding_fraction (R²) | 0.623 | 0.617 | −0.006 |
| gene_density (R²) | 0.699 | 0.693 | −0.006 |
| noncoding_bp (R²) | 0.696 | 0.695 | −0.001 |
| n_genes (R²) | 0.825 | 0.828 | +0.003 |
| mean_intergenic_len (R²) | 0.364 | 0.373 | +0.010 |

Todas as diferenças ≤ 0.01 (dentro do DP entre folds) → sem leakage de quase-duplicatas; RefSeq é não-redundante por design.
