#!/usr/bin/env python3
"""make_supplementary.py — monta o Material Suplementar UNICO da revisao R1.

As tabelas sao renumeradas pela ORDEM DE CITACAO no manuscrito revisado, que e o que a
Frontiers exige. A ordem antiga (S1-S4 da submissao + S5-S8 da revisao) NAO era a de
aparicao: a tabela de selecao de amostras, criada por ultimo, e a segunda a ser citada.

Nada e digitado: os numeros das quatro tabelas da submissao sao lidos do
`supplementary_tables.md` do repositorio de reprodutibilidade, e os da revisao vem dos
JSONs cacheados e das tabelas geradas em results/tables/. As legendas sao texto.

Uso:
    python code/05_figures/make_supplementary.py \
        --out results/tables/supplementary_material_R1.md
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def md_section(path, header_startswith):
    """Devolve o corpo de uma secao markdown (## ...) de um arquivo, sem o cabecalho."""
    txt = open(path).read().split("\n")
    out, on = [], False
    for l in txt:
        if l.startswith("## "):
            if on:
                break
            on = l.startswith("## " + header_startswith) or header_startswith in l
            continue
        if on:
            out.append(l)
    if not out:
        sys.exit(f"secao '{header_startswith}' nao encontrada em {path}")
    return "\n".join(out).strip()


def table_only(body):
    """So as linhas da tabela markdown, descartando prosa da secao."""
    rows = [l for l in body.split("\n") if l.strip().startswith("|")]
    if not rows:
        sys.exit("sem tabela no bloco")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", default=ROOT, help="raiz do repositorio com as tabelas da submissao")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables",
                                                  "supplementary_material_R1.md"))
    a = ap.parse_args()
    SUB = os.path.join(a.repro, "results", "tables", "supplementary_tables.md")
    T = lambda f: os.path.join(ROOT, "results", "tables", f)
    J = lambda f: json.load(open(os.path.join(ROOT, "results", "json", f)))

    o = []
    o.append("**Supplementary Material**\n")
    o.append("*Decoding taxonomy and genome-level architecture from Evo 2 embeddings of viral "
             "genomes: a linear-probe benchmark against compositional baselines*\n")
    o.append("Tables are numbered in order of first citation in the manuscript.\n")
    o.append("**1 Supplementary Tables**\n")

    # ---- S1: composicao do corpus (era S4) -- citada em Methods 2.1
    o.append("**Supplementary Table S1.** Composition of the viral genome corpus (n = 19,429 "
             "RefSeq records, including individual segments of segmented viruses), "
             "cross-tabulated by Baltimore replication class and host domain as assigned "
             "through the ICTV Virus Metadata Resource (MSL41). Records whose family- or "
             "genus-level taxonomy could not be mapped to a Baltimore class or to a host domain "
             "in the VMR are reported as unassigned/unknown. Probe subsets were drawn as "
             "balanced samples from these quota groups, so that the strong imbalance of the full "
             "corpus — for example, 6,311 class I records versus 94 class VI — does not "
             "propagate into the probes.\n")
    o.append(table_only(md_section(SUB, "Table S4.")) + "\n")

    # ---- S2: selecao de amostras (era S8) -- Methods 2.1
    sel = open(T("supp_S8_selection.md")).read()
    caption = re.search(r"^Composition of every analysed population\..*?$", sel, re.M).group(0)
    o.append("**Supplementary Table S2.** " + caption.replace(
        "This table replaces the flow diagram requested in review and resolves",
        "It resolves") + "\n")
    o.append(table_only(sel) + "\n")
    for extra in re.findall(r"^\*\*(?:Segmented viruses|Records without a family assignment).*$",
                            sel, re.M):
        o.append(extra + "\n")

    # ---- S3: controle de precisao (era S1) -- Methods 2.5
    o.append("**Supplementary Table S3.** Precision control on the NVIDIA H100. Cross-validated "
             "probe performance (repeated cluster-aware cross-validation, mean ± SD) for the "
             "Evo 2 20B embedding extracted with native FP8 input projections versus the same "
             "layers re-extracted with FP8 disabled (forced bfloat16), on identical genomes and "
             "folds. Classification reports accuracy; regression reports R². Forcing bfloat16 "
             "degrades every target, most severely the fine-grained architectural features, "
             "indicating that the 20B checkpoint is calibrated for FP8 inference rather than "
             "precision-agnostic.\n")
    o.append(table_only(md_section(SUB, "Table S1.")) + "\n")

    # ---- S4: CV aleatoria vs cluster-aware (era S2)
    o.append("**Supplementary Table S4.** Effect of sequence-identity-aware cross-validation. "
             "Probe performance under random repeated cross-validation versus group-aware "
             "cross-validation in which MMseqs2 linclust clusters (95% identity, 85% coverage) "
             "are used as groups, so that no cluster spans training and held-out folds. Δ is the "
             "change induced by the group-aware scheme. Only 16 of the 1,912 probed genomes "
             "merged into clusters, and no metric changes by more than 0.012 in either model. "
             "Identity clustering addresses near-duplicate leakage only; phylogenetic dependence "
             "is addressed by the family-grouped analyses of Supplementary Tables S6 and S7.\n")
    o.append(table_only(md_section(SUB, "Table S2.")) + "\n")

    # ---- S5: classificacao + recall por familia (era S7, ampliada)
    o.append("**Supplementary Table S5.** Classification beyond accuracy, under the "
             "pre-registered identity-clustered scheme. Balanced accuracy and macro-F1 are "
             "reported because the classes are unbalanced.\n")
    o.append(table_only(md_section(T("supp_S5_S7.md"), "Table S7.")) + "\n")
    cls = J("classification_metrics.json")["targets"]["family"]["reps"]
    o.append("**Per-family recall** (Evo 2 20B `blocks.18`, same folds):\n")
    o.append("| Family | n | Recall |")
    o.append("|---|---|---|")
    per = cls["evo2_20b_blocks18"]["cl95"]["per_class_recall"]
    for fam, v in sorted(per.items(), key=lambda kv: -kv[1]["recall"]):
        o.append(f"| *{fam}* | {v['n']} | {v['recall']:.3f} |")
    o.append("")

    # ---- S6: contrastes primarios (era S5)
    o.append("**Supplementary Table S6.** Primary confirmatory contrasts under both "
             "cross-validation schemes. Evo 2 20B `blocks.18` versus the 6-mer baseline on the "
             "six pre-declared architecture targets. ΔR² is the difference in cross-validated "
             "R²; the test is a Nadeau–Bengio corrected resampled t-test and the interval is "
             "obtained by inverting the same statistic. p-values are Holm-corrected within the "
             "set of six.\n")
    o.append(table_only(md_section(T("supp_S5_S7.md"), "Table S5.")) + "\n")

    # ---- S7: melhor baseline por classe (era S6)
    body = md_section(T("supp_S5_S7.md"), "Table S6.")
    o.append("**Supplementary Table S7.** Evo 2 versus the strongest baseline of each class. For "
             "each target and scheme, the best baseline within each class was selected and "
             "contrasted with the embedding. Compositional: k-mer (k = 3–6), multi-k, codon, "
             "dicodon, GC-and-length. Annotation-derived: the six-frame ORF scan and the "
             "combined representation containing it. A positive ΔR² favours Evo 2.\n")
    o.append(table_only(body) + "\n")
    o.append(re.search(r"^\*\*Summary\.\*\*.*$", body, re.M).group(0) + "\n")

    # ---- S8: sensibilidade da definicao de gene overlap (NOVA)
    ov = J("overlap_sensitivity.json")
    o.append("**Supplementary Table S8.** Sensitivity of the gene-overlap contrast to the "
             f"definition of the feature (n = {ov['n']:,}). The published definition counts a "
             "position once per overlapping CDS pair and does not separate strands; "
             "*positions* counts genome positions covered by two or more CDS features; "
             "*same-strand* restricts to overlap between features on the same strand. Evo 2 20B "
             "`blocks.18` versus the 6-mer baseline. The conclusion holds under all three "
             "definitions in both schemes.\n")
    o.append("| Definition | Scheme | Evo 2 R² | 6-mer R² | ΔR² | p |")
    o.append("|---|---|---|---|---|---|")
    NICE = {"published": "Published", "positions": "Positions covered", "same_strand": "Same strand"}
    SCH = {"cl95": "Identity-clustered (pre-registered)", "family": "Family-grouped"}
    for d, per in ov["results"].items():
        for sch, r in per.items():
            o.append(f"| {NICE[d]} | {SCH[sch]} | {r['r2_evo2']:.3f} | {r['r2_6mer']:.3f} | "
                     f"{r['delta']:+.3f} | {fmt_p(r['p'])} |")
    o.append("")

    # ---- S9: PCA-150 (era S3)
    o.append("**Supplementary Table S9.** Dimensionality-matched comparison. Because the 20B "
             "embedding is higher-dimensional than the 7B (8,192 versus 4,096 components), which "
             "could by itself inflate linear-probe performance, every probe was re-run with both "
             "embeddings projected onto a common 150-component PCA subspace, fitted inside each "
             "cross-validation training fold to prevent leakage. The number of components "
             "follows from min(n_train, n_features), with approximately 279 training samples per "
             "fold for the family target. All values are cluster-aware cross-validation, "
             "mean ± SD.\n")
    o.append(table_only(md_section(SUB, "Table S3.")) + "\n")

    # ---- S10: generalizacao por familia (LOFO + dentro de familia)
    fam = J("family_cv_metrics.json")
    wf = J("within_family_cv_metrics.json")
    o.append("**Supplementary Table S10.** Family-level generalisation. Leave-one-family-out is "
             "restricted to families with at least 30 genomes and summarised by R² over pooled "
             "out-of-fold predictions. Within-family cross-validation is run inside each such "
             "family and summarised by the median across families; it separates family-level "
             "from within-family signal, which leave-one-family-out conflates. SD ratio is the "
             "median within-family standard deviation of the target divided by its global "
             "standard deviation — a low ratio means restriction of range, which bounds the "
             "attainable R² regardless of representation.\n")
    o.append("| Target | LOFO pooled R² (Evo 2) | Within-family median R² (Evo 2) | "
             "Within-family median R² (6-mer) | Families | Median SD ratio |")
    o.append("|---|---|---|---|---|---|")
    import statistics as st
    NICE_T = {"coding_fraction": "Coding fraction", "gene_density": "Gene density",
              "noncoding_bp": "Non-coding bp", "n_genes": "Gene count",
              "mean_intergenic_len": "Mean intergenic length", "overlap_bp": "Gene overlap"}
    for t, nice in NICE_T.items():
        lofo = fam["targets"][t]["reps"]["evo2_20b_blocks18"].get("lofo", {}).get("pooled_r2")
        fams = wf["targets"][t]["families"]
        e = st.median([f["reps"]["evo2_20b_blocks18"]["r2"] for f in fams.values()])
        b = st.median([f["reps"]["6mer"]["r2"] for f in fams.values()])
        sd = st.median([f["sd_within"] / wf["targets"][t]["sd_global"] for f in fams.values()])
        lofo_s = f"{lofo:.3f}" if isinstance(lofo, (int, float)) else "—"
        o.append(f"| {nice} | {lofo_s} | {e:.3f} | {b:.3f} | {len(fams)} | {sd:.2f} |")
    o.append("")

    open(a.out, "w").write("\n".join(o) + "\n")
    print(f"-> {a.out}")


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else (f"{p:.2g}" if p < 0.1 else f"{p:.2f}")


if __name__ == "__main__":
    main()
