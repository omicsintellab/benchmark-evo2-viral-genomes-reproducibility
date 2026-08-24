## Supplementary Table S8. Sample selection

Composition of every analysed population. Counts are derived from the released data files, not transcribed. This table replaces the flow diagram requested in review and resolves the apparent inconsistency between 981, 1,080 and 1,912.

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

