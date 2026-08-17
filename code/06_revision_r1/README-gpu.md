# Revision R1 — GPU scripts

Scripts for the GPU stage of the R1 revision. They are self-contained: copy this directory
to the GPU host and run: nothing here imports from the rest of the repository.

| Script | Answers | Cost |
|---|---|---|
| `00_preflight.py` | nothing — it is the gate that decides whether a given GPU can run the model at all | minutes |
| `01_block_shuffle.py` | R1 3.3 (gene order / genome arrangement) | hours |
| `02_generation_rerun.py` | R1 4.2, R2 #11/#12 (generation metrics) | hours |
| `evo2_compat.py` | shared helpers (`patch_fp8`, `make_windows`, weight resolution, sharding) | — |

> Paths are placeholders: `<WEIGHTS>` is the directory holding the Evo 2 checkpoint and
> `<SCRATCH>` a large scratch mount. The scripts take both on the command line.

## Run the preflight first

The 20B checkpoint is ~40 GB in bf16 and the analysis scripts are single-GPU
(`--device cuda:0`): the original run used **one H100 80 GB**. On any smaller card the model
does not fit, and discovering that after two hours of embedding costs the whole run.
`00_preflight.py` loads the model under several strategies, each **in its own subprocess**,
and probes peak memory at 4k/8k/16k/32k context.

The subprocess isolation is not fastidiousness. A load that fails with OOM leaves the partial
model referenced by the traceback; `gc.collect()` and `empty_cache()` do not return the VRAM,
and the next strategy silently spills onto a second GPU. Single-GPU strategies additionally
run with `CUDA_VISIBLE_DEVICES=0` so that a "success" cannot be a disguised spill.

```bash
python 00_preflight.py --weights-local <WEIGHTS> --out out/preflight_report.json
```

`--weights-local` accepts either the `.pt` file or a root containing `<model>/<model>.pt`.
Read `verdict.recommended` from the JSON, then run 01/02 with that model.

## Estado: esperando uma GPU única de ≥80 GB (`p5`)

Duas máquinas multi-GPU foram tentadas e **abandonadas**. Rode numa `p5` (H100 80 GB), que é
o hardware do artigo: aí o caminho single-GPU funciona e **não se usa `--pipeline`**.

| Máquina | 20B | 7B |
|---|---|---|
| 4× L4 (`g6.12xlarge`, 22 GB/card) | não carrega em card nenhum | só até 8k |
| 4× L40S (`g6e.12xlarge`, 44,5 GB/card) | single só até **4k**; fatiado até **16k**, OOM em 32k | 32k OK |

O artigo usa janela de 32k, então o embaralhamento de blocos no 20B não fecha em nenhuma das
duas. A geração (5.120 tokens) **rodou** no 20B fatiado e passou no controle positivo, mas a
75–77 s por geração — o fatiamento em pipeline é sequencial, só uma GPU trabalha por vez, e
batching não ajuda porque o trabalho por hop cresce com o lote. Corrida cheia: 25 h.

**Vale para a `p5` também:** `generate()` vaza estado entre chamadas (cache de inferência
preso ao modelo; a GPU 0 chegou a 39,5 GiB com ~12 GiB de pesos). Isso derruba corridas longas
em qualquer hardware. Já corrigido em `02_generation_rerun.py`.

## What we learned on 4× NVIDIA L4 (`g6.12xlarge`), August 2026

This configuration was tried because no ≥80 GB instance was available. **It does not work for
the 20B**, and the attempt was stopped. Recorded here so nobody repeats it:

| Strategy | Result |
|---|---|
| 20B, single GPU | **OOM on load.** Each L4 exposes 21.95 GiB usable; weights are ~40 GB. |
| 20B, `accelerate.dispatch_model` | **Loads** (40 GB does fit across 88 GB) but the forward dies: `Expected all tensors to be on the same device`. |
| 7B, single GPU | Works to **8k context** (peak 16.78 GB); OOM at 16k. The paper uses 32k windows. |
| 20B/7B, manual pipeline sharding | Implemented (`evo2_compat.shard_pipeline`) but **never validated** — the run was stopped first. Treat as untested. |

Why `dispatch_model` fails: it installs hooks that move each submodule's *inputs* to that
submodule's device, which assumes all data flow passes through `forward` signatures.
StripedHyena does not satisfy that assumption. `shard_pipeline` was written to sidestep it by
splitting the block list explicitly and moving the hidden state by hand, but it rests on the
assumption that only the hidden state crosses blocks — if any tensor is held as a loose
attribute (neither parameter nor registered buffer), it will not move. `describe_devices`
exists to locate exactly that, should anyone pick this up again.

**On a single ≥80 GB GPU none of this is needed** — the single-GPU path is the one the paper
used and the one to prefer.

FP8 is not the obstacle: L4, L40S and H100 are all cc ≥ 8.9 and `patch_fp8` keeps FP8 on, so
embeddings stay comparable to the cached ones. `--no-fp8` exists for diagnosis only; bf16
changes the embeddings measurably (see Supplementary Table S1) and would have to be declared.

## Scratch space

On a typical GPU host the home directory is a small volume and the 20B weights alone are
~48 GB. Put weights, inputs, outputs **and caches** on the large NVMe mount:

```bash
export R1=<SCRATCH>/r1
export HF_HOME=$R1/cache/hf HUGGINGFACE_HUB_CACHE=$R1/cache/hf
export TORCH_HOME=$R1/cache/torch XDG_CACHE_HOME=$R1/cache TMPDIR=$R1/tmp
mkdir -p $R1/{scripts,inputs,out,weights,tmp,cache}
```

The caches matter as much as the weights: without redirecting `HF_HOME`/`TORCH_HOME` the
config and tokenizer downloads land in `~/.cache` and fill the EBS volume mid-run; without
`TMPDIR`, `.npz` compression and `pip` builds do the same. That mount is ephemeral — sync
results out before stopping the instance.

## Block-shuffle design

`01_block_shuffle.py` permutes the **order** of contiguous blocks: global composition is
identical, local composition nearly so (only the joins change), long-range arrangement is
destroyed. The `--blocks` sweep gives a dose-response rather than a single point.

Original and shuffled genomes are embedded **in the same run**. This is essential: comparing
shuffled embeddings against a previously cached set would confound rearrangement with
precision and hardware.

Interpretation fixed **before** seeing the result: if the embedding barely moves under block
permutation, Evo 2 does not represent long-range arrangement and the claim narrows to
genome-level architecture-associated summary features — the outcome R1 3.3 already permits.
If it moves well beyond noise, arrangement is represented, supported by direct evidence
rather than an indirect probe.

## Generation re-run

`02_generation_rerun.py` exists because the original script consumed generated sequences in
memory and stored only bits/nt and 4-mer cosine. Sequence identity, edit distance, ORF
continuity, frameshift and premature stops — all of which R2 #12 asks for — are simply not
computable from what was kept. This script persists the generated sequences, with replicates
per genome and a temperature/top-k sweep, reusing the same accessions and `prompt_bp`/`gap_bp`
as the original run so the comparison stays clean. It also writes the **true** gap sequences;
without them nothing can be compared downstream.
