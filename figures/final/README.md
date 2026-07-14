# Figures as submitted

These are the three figures of the paper, numbered as they appear in the manuscript. They are byte-identical in content to the generated figures in the parent directory — only renamed.

| File | Content | Generated as |
|---|---|---|
| `figure1.{png,svg}` | Representation probes: Baltimore confusion matrix, classification accuracy, per-feature R², PCA projection | `figure1_combined` |
| `figure2.{png,svg}` | Generative evaluation: genome perplexity, completion likelihood and composition vs a Markov baseline | `figure3_20b` |
| `figure3.{png,svg}` | Representation quality by model scale and network depth: layer sensitivity, 20B vs 7B at matched dimensionality | `figure3_combined` |

**Why this directory exists.** The generated file names follow the script that produced them, not the paper's numbering — `figure3_20b` is *Figure 2*, and `figure3_combined` is *Figure 3*. That mismatch is easy to get wrong when assembling a submission package (it once was), so the figures are archived here under the numbering that the legends actually refer to.

The TIFF versions required by the journal are not kept here: they are deterministic 300-dpi renderings of these PNGs and add ~29 MB. Regenerate with Pillow if needed:

```python
from PIL import Image
for n in (1, 2, 3):
    Image.open(f"figure{n}.png").save(f"figure{n}.tiff", compression="tiff_lzw", dpi=(300, 300))
```
