#!/usr/bin/env python3
"""final_statistics.py — a estatística final da revisão (R1 3.5, R2 #6).

Não recomputa CV nenhuma: lê os scores de fold já cacheados em
`family_cv_metrics.json` e `composition_baselines_metrics.json` e produz o bloco
estatístico definitivo. Isso importa porque o esquema PRIMÁRIO mudou depois das corridas
(voltou a ser o pré-registrado `cl95`, ver PLANO §4.4-bis) e os scores dos quatro esquemas
já estavam salvos — refazer a CV seria desperdício e introduziria variação de semente.

O que os revisores pedem e sai daqui:
  - R1 3.5 / R2 #6: teste apropriado para CV repetida (Nadeau-Bengio, que corrige a
    variância pelo fator (1/n + n_test/n_train)), com IC bootstrap e tamanho de efeito
    ao lado do p, e correção de comparações múltiplas dentro de um conjunto primário
    pequeno e declarado.
  - Conjunto primário: 6 contrastes Evo 2 20B blocks.18 vs 6-mer, sob `cl95`.
  - Controle negativo pré-declarado (cpg_oe/upa_oe), FORA da correção.
  - Sensibilidade: os mesmos contrastes sob agrupamento por família.

Uso:
    python final_statistics.py --out ../../results/json
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from family_cv import PRIMARY, NEGCTRL, nadeau_bengio, boot_ci, holm, SPLITS

ROOT = os.path.join(HERE, "..", "..")


def cohens_d(a, b):
    """Tamanho de efeito para observações pareadas (d de Cohen para diferenças)."""
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def contrast(e, b, label_a, label_b):
    t, p = nadeau_bengio(e, b, SPLITS)
    lo, hi = boot_ci(e, b)
    return {"a": label_a, "b": label_b,
            "r2_a": float(np.mean(e)), "r2_b": float(np.mean(b)),
            "delta": float(np.mean(e) - np.mean(b)), "ci95": [lo, hi],
            "cohens_d": cohens_d(e, b), "t_nadeau_bengio": t, "p_raw": p,
            "n_folds": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "json"))
    a = ap.parse_args()

    fam = json.load(open(os.path.join(ROOT, "results", "json", "family_cv_metrics.json")))
    comp_path = os.path.join(ROOT, "results", "json", "composition_baselines_metrics.json")
    comp = json.load(open(comp_path)) if os.path.exists(comp_path) else None

    EVO, KMER = "evo2_20b_blocks18", "6mer"
    out = {"design": {
        "primary_scheme": "cl95",
        "primary_scheme_rationale":
            "Esquema PRE-REGISTRADO (corpus_design.yaml: dedup min_seq_id 0.95, "
            "split cluster_aware). Agrupar por familia responde outra pergunta "
            "(independencia taxonomica) e nao era o desenho fixado antes dos dados; "
            "promove-lo a principal seria a mesma selecao pos-hoc que R1 3.5 critica.",
        "primary_set": [n for n, _ in PRIMARY],
        "contrast": f"{EVO} vs {KMER}",
        "test": "t reamostrado corrigido (Nadeau & Bengio, 2003), fator (1/n + 1/(k-1))",
        "correction": "Holm dentro do conjunto primario de 6",
        "ci": "IC obtido invertendo a MESMA estatistica corrigida do teste "
               "(ver ci_consistency.json); o bootstrap sobre diferencas por fold trata os "
               "folds como independentes e foi abandonado",
        "negative_control": [n for n, _ in NEGCTRL],
        "negative_control_note":
            "Alvos de composicao de dinucleotideo, declarados ANTES como casos em que se "
            "espera que o 6-mer vença. Nao sao testes de hipotese: sao controle de "
            "especificidade do probe. FORA da correcao.",
        "exploratory":
            "Baseline GC+len; classificacao de Baltimore/hospedeiro/familia; varredura de "
            "camadas; 20B vs 7B; controle de precisao; controle de PCA; CV aleatoria; "
            "avaliacao de geracao. Reportados com IC, sem p confirmatorio.",
    }, "primary": {}, "sensitivity_family": {}, "negative_control": {}}

    # ---- primário: cl95
    pv, names = [], []
    for name, _ in PRIMARY:
        r = fam["targets"][name]["reps"]
        c = contrast(r[EVO]["cl95"]["scores"], r[KMER]["cl95"]["scores"], EVO, KMER)
        out["primary"][name] = c
        pv.append(c["p_raw"]); names.append(name)
    adj = holm(pv, names)
    for name in names:
        out["primary"][name]["p_holm"] = adj[name]
        out["primary"][name]["significant_holm"] = bool(adj[name] < 0.05 and
                                                        out["primary"][name]["delta"] > 0)

    # ---- sensibilidade: mesmos contrastes sob agrupamento por família
    pv2 = []
    for name, _ in PRIMARY:
        r = fam["targets"][name]["reps"]
        c = contrast(r[EVO]["family"]["scores"], r[KMER]["family"]["scores"], EVO, KMER)
        out["sensitivity_family"][name] = c
        pv2.append(c["p_raw"])
    adj2 = holm(pv2, names)
    for name in names:
        out["sensitivity_family"][name]["p_holm"] = adj2[name]

    # ---- controle negativo, sem correção
    for name, _ in NEGCTRL:
        r = fam["targets"][name]["reps"]
        out["negative_control"][name] = {
            "cl95": contrast(r[EVO]["cl95"]["scores"], r[KMER]["cl95"]["scores"], EVO, KMER),
            "family": contrast(r[EVO]["family"]["scores"], r[KMER]["family"]["scores"],
                               EVO, KMER)}

    # ---- baseline mais forte por classe (etapa 3), se disponível
    if comp:
        out["vs_best_baseline_by_class"] = {
            sch: {t: {cls: {k: v[k] for k in ("best", "best_r2", "delta", "ci95", "p",
                                              "evo2_ahead")}
                      for cls, v in tv["by_class"].items()}
                  for t, tv in comp["tests"][sch].items()}
            for sch in ("cl95", "family")}
        out["design"]["baseline_classes"] = comp["classes"]

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "final_statistics.json")
    json.dump(out, open(dst, "w"), indent=1)

    def row(n, c, pkey):
        ci = f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
        print(f"{n:<22}{c['r2_a']:>8.3f}{c['r2_b']:>8.3f}{c['delta']:>8.3f}{ci:>16}"
              f"{c['cohens_d']:>7.2f}{c[pkey]:>10.2g}")

    hdr = f"\n{'alvo':<22}{'Evo2':>8}{'6-mer':>8}{'dR2':>8}{'IC95':>16}{'d':>7}{'p':>10}"
    print(hdr)
    print("-- PRIMARIO (cl95, pre-registrado), p corrigido por Holm " + "-" * 18)
    for n in names:
        row(n, out["primary"][n], "p_holm")
    print("-- SENSIBILIDADE (agrupado por familia), p Holm " + "-" * 27)
    for n in names:
        row(n, out["sensitivity_family"][n], "p_holm")
    print("-- CONTROLE NEGATIVO (fora da correcao), p bruto " + "-" * 26)
    for n, _ in NEGCTRL:
        row(n, out["negative_control"][n]["cl95"], "p_raw")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
