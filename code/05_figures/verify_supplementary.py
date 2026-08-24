#!/usr/bin/env python3
"""verify_supplementary.py — read the Supplementary .docx (READ-ONLY) and check every cell
against the artefact that produced it: the JSONs in results/json/ and the tables in
results/tables/ of this repository. Nothing is rewritten. The .docx is not versioned here: use
the Supplementary Material published by the journal, or the one built by make_supplementary.py.

The COVERAGE guard at the end exists because an earlier version of this script compared the
WRONG table through an index shift and passed silently: zero divergences because zero
comparisons. Each table declares a minimum number of checks.

Usage:  python code/05_figures/verify_supplementary.py --docx <Supplementary_Material.docx>
"""
import argparse, json, os, re, sys, zipfile, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--docx", required=True,
                help="Supplementary Material .docx (the submitted artefact; not versioned here)")
ap.add_argument("--root", default=ROOT, help="root of this repository")
args = ap.parse_args()
R = P = args.root
J = lambda f: json.load(open(f"{R}/results/json/{f}"))

x = zipfile.ZipFile(args.docx).read("word/document.xml").decode()
# tabelas -> lista de linhas -> lista de cells (texto)
tables = []
for tbl in re.findall(r"<w:tbl>.*?</w:tbl>", x, re.S):
    rows = []
    for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S):
        cells = ["".join(re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", tc, re.S)).strip()
                 for tc in re.findall(r"<w:tc>(?:(?!<w:tc>).)*?</w:tc>", tr, re.S)]
        rows.append(cells)
    tables.append(rows)
print(f"tables read from the docx: {len(tables)}")

fails, checks, COVER = [], 0, []
def eq(label, got, exp, tol=0.0011):
    global checks
    checks += 1
    COVER.append(label)
    try:
        ok = abs(float(got) - float(exp)) <= tol
    except (TypeError, ValueError):
        ok = str(got).strip() == str(exp).strip()
    if not ok:
        fails.append(f"{label}: docx={got!r} artefato={exp!r}")

def num(s):
    s = s.replace(",", "").replace("−", "-").replace("+", "")
    m = re.match(r"^-?\d+\.?\d*", s.strip())
    return float(m.group(0)) if m else None

# ---------- S1: corpus composition (row and column sums)
t1 = tables[0]
tot_row = [r for r in t1 if r and r[0].strip().startswith("**Total")or r[0].strip()=="Total"]
body = [r for r in t1[1:] if r and r[0] and not r[0].strip().startswith("Total") and "**Total" not in r[0]]
col_sums = [0]*5
for r in body:
    vals = [num(c) for c in r[1:6]]
    if None in vals: continue
    eq(f"S1 row sum {r[0]}", sum(vals[:4]), vals[4])
    for i,v in enumerate(vals): col_sums[i]+=v
if tot_row:
    tv=[num(c) for c in tot_row[0][1:6]]
    for i,lbl in enumerate(["Eukaryote","Bacteria","Archaea","Unknown","Total"]):
        eq(f"S1 column total {lbl}", col_sums[i], tv[i])
eq("S1 grand total = 19,429", col_sums[4], 19429)

# ---------- S2: sample selection (declared arithmetic)
t2 = tables[1]
g = {r[0].strip(): num(r[1]) for r in t2[1:] if len(r)>1 and num(r[1]) is not None}
eq("S2 Baltimore 981", g.get("Baltimore probe subset"), 981)
eq("S2 union 1,912 = 981+1200-269", g.get("Union of both subsets"), 981+1200-269)
eq("S2 host 1,080 = 1200-120", g.get("Host-domain classification"), 1200-120)
eq("S2 family 1,691 = 1912-221", g.get("Family-grouped analyses"), 1912-221)

# ---------- S5: classification + recall
cls = J("classification_metrics.json")["targets"]
t5, t5b = tables[4], tables[5]
for r in t5[1:]:
    if len(r) < 7: continue
    tgt, rep = r[0].strip(), r[3].strip().strip("`")
    v = cls[tgt]["reps"][rep]["cl95"]
    eq(f"S5 {tgt}/{rep} accuracy", num(r[4]), v["accuracy"])
    eq(f"S5 {tgt}/{rep} balanced", num(r[5]), v["balanced_accuracy"])
    eq(f"S5 {tgt}/{rep} macroF1", num(r[6]), v["macro_f1"])
per = cls["family"]["reps"]["evo2_20b_blocks18"]["cl95"]["per_class_recall"]
for r in t5b[1:]:
    if len(r) < 3: continue
    fam = r[0].strip().strip("*")
    if fam in per:
        eq(f"S5 recall {fam}", num(r[2]), per[fam]["recall"])
        eq(f"S5 n {fam}", num(r[1]), per[fam]["n"])

# ---------- S6: primary contrasts
ci = J("ci_consistency.json")["schemes"]
SCH = {"Identity-clustered (pre-registered)": "cl95", "Family-grouped": "family"}
NICE = {"Coding fraction":"coding_fraction","Gene density":"gene_density","Non-coding bp":"noncoding_bp",
        "Gene count":"n_genes","Mean intergenic length":"mean_intergenic_len","Gene overlap":"overlap_bp"}
for r in tables[5+1][1:]:
    if len(r) < 6 or r[0].strip() not in SCH: continue
    sch, tgt = SCH[r[0].strip()], NICE.get(r[1].strip())
    if not tgt: continue
    v = ci[sch][tgt]
    eq(f"S6 {sch}/{tgt} delta", num(r[2]), v["delta"])
    lo, hi = re.findall(r"[-+−]?\d+\.\d+", r[3].replace("−","-"))[:2]
    eq(f"S6 {sch}/{tgt} CI lo", float(lo), v["ci95_nb"][0], tol=0.0015)
    eq(f"S6 {sch}/{tgt} CI hi", float(hi), v["ci95_nb"][1], tol=0.0015)
    eq(f"S6 {sch}/{tgt} d", num(r[4]), v["cohens_d"], tol=0.06)

# ---------- S8: overlap sensitivity
ov = J("overlap_sensitivity.json")["results"]
DEF = {"Published":"published","Positions covered":"positions","Same strand":"same_strand"}
for r in tables[8][1:]:
    if len(r) < 6 or r[0].strip() not in DEF: continue
    d, sch = DEF[r[0].strip()], SCH[r[1].strip()]
    v = ov[d][sch]
    eq(f"S8 {d}/{sch} evo2", num(r[2]), v["r2_evo2"])
    eq(f"S8 {d}/{sch} 6mer", num(r[3]), v["r2_6mer"])
    eq(f"S8 {d}/{sch} delta = evo2-6mer", num(r[4]), v["r2_evo2"]-v["r2_6mer"], tol=0.0015)

# ---------- S10: LOFO + within-family
fam = J("family_cv_metrics.json")["targets"]; wf = J("within_family_cv_metrics.json")["targets"]
for r in tables[10][1:]:
    if len(r) < 6: continue
    tgt = NICE.get(r[0].strip())
    if not tgt: continue
    lofo = fam[tgt]["reps"]["evo2_20b_blocks18"].get("lofo", {}).get("pooled_r2")
    fams = wf[tgt]["families"]
    eq(f"S10 {tgt} LOFO", num(r[1]), lofo)
    eq(f"S10 {tgt} within evo2", num(r[2]), st.median([f["reps"]["evo2_20b_blocks18"]["r2"] for f in fams.values()]))
    eq(f"S10 {tgt} within 6mer", num(r[3]), st.median([f["reps"]["6mer"]["r2"] for f in fams.values()]))
    eq(f"S10 {tgt} n families", num(r[4]), len(fams))
    eq(f"S10 {tgt} sd ratio", num(r[5]),
       st.median([f["sd_within"]/wf[tgt]["sd_global"] for f in fams.values()]), tol=0.006)

print(f"\nnumeric checks: {checks} | divergences: {len(fails)}")
for f in fails: print("  X", f)

# ================== SECOND PASS: S3, S4, S7, S9 ==================
print("\n--- S3/S4/S9 against results/tables/supplementary_tables.md ---")
sub = open(f"{P}/results/tables/supplementary_tables.md").read()
def md_rows(header):
    blk = sub.split(header)[1].split("## ")[0]
    return [ [c.strip() for c in l.strip().strip("|").split("|")]
             for l in blk.split("\n") if l.strip().startswith("|") and "---" not in l ]

for docx_idx, header, label in ((2, "Table S1.", "S3"), (3, "Table S2.", "S4"), (9, "Table S3.", "S9")):
    src = md_rows(header)
    got = tables[docx_idx]
    src_body = src[1:]
    got_body = [r for r in got[1:] if any(c.strip() for c in r)]
    eq(f"{label} number of rows", len(got_body), len(src_body))
    for gr, sr in zip(got_body, src_body):
        for gc, sc in zip(gr[1:], sr[1:]):
            a, b = num(gc), num(sc)
            if a is not None and b is not None:
                eq(f"{label} row '{sr[0][:22]}'", a, b)

print("\n--- S7 against ci_consistency.vs_best_baseline_by_class ---")
vb = J("ci_consistency.json")["vs_best_baseline_by_class"]
CLS = {"Compositional": "compositional", "Annotation-derived": "annotation_derived"}
for r in tables[7][1:]:
    if len(r) < 8 or r[0].strip() not in SCH: continue
    sch, tgt, cls_ = SCH[r[0].strip()], NICE.get(r[1].strip()), CLS.get(r[2].strip())
    if not (tgt and cls_): continue
    v = vb[sch][tgt][cls_]
    eq(f"S7 {sch}/{tgt}/{cls_} best baseline", r[3].strip().strip("`"), v["best"])
    eq(f"S7 {sch}/{tgt}/{cls_} baseline R2", num(r[4]), v["best_r2"])
    eq(f"S7 {sch}/{tgt}/{cls_} delta", num(r[5]), v["delta"])

print(f"\nTOTAL: {checks} checks | {len(fails)} divergences")
for lbl, minimo in (("S1 ",6),("S2 ",4),("S3 ",8),("S4 ",8),("S5 ",12),("S6 ",24),("S7 ",24),("S8 ",18),("S9 ",8),("S10 ",30)):
    n=sum(1 for c in COVER if c.startswith(lbl))
    print(("  OK  " if n>=minimo else "  LOW COVERAGE ")+f"{lbl.strip()}: {n} checks (min {minimo})")
for f in fails: print("  X", f)
