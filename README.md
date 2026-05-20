# Attention Compression

Compression experiments around **`allenai/OLMo-1B-0724-hf`**, using the Dolma **`v1_6-sample`** corpus as a reproducible slice of the model's data distribution.

The core idea is **asymmetric compression inside attention**:

- **Q/K “routing”** often lives in empirical subspaces that are **much smaller than the nominal head geometry** suggests.
- **V and the residual after attention** behave more like **broad payloads**—they tolerate less aggressive shrinkage without visible damage.

The repository is intentionally split:

- **Root (`README.md` here)** — stable overview: what we're doing now and how we got here.
- **[`research/`](research/)** — long runs, numeric tables, and superseded hypotheses ([`research/FINDINGS.md`](research/FINDINGS.md)).

---

## Where we landed (current “state of the art” in this codebase)

Roughly validated outcomes (always check [`research/FINDINGS.md`](research/FINDINGS.md) for citations and caveats):

1. **Q/K-first compression works when V stays full.** Co-trained low-rank Q/K branches (with dense V per head) can match teacher head outputs and attention behavior far better than naively chopping the score dimension alone.
2. **V compression is the first cliff.** Moving from full V (`128`) toward `96`/`64` in the ladders clusters quality loss in a repeatable way—not just “a bit more MSE everywhere.”
3. **Whole-model deployment is imaginable but not free.** Replacing **all heads’ Q/K** with low-rank factors + dense **V**, then optional **short Q/K-only fine-tuning**, lands in modest **perplexity regressions on stratified window eval**, not catastrophe.
4. **Inference parity needs engineering**, not only math. Pure Python low-rank was slow on decode; fused paths and optionally **dense materialization** of live Q/K bring throughput back toward vanilla GEMMs (see **`attention_compression.qk_surgery`** and **`scripts/21_inference_speed_benchmark.py`**).
5. **Downstream bottleneck path:** capture layer internals (“heads before **`o_proj`**, FFN input”), train bottleneck / mimic variants, then **co-train a smaller FFN with a mimic **`o_proj`**** (**`scripts/32_train_bottleneck_ffn_after_mimic_oproj.py`**). **Relative-only MSE objectives can diverge badly in residual direction on some layers**; **cosine** or **`both(w)` losses keep eval cosine near 1** while trading off `relative_mse` via **`w`** ( **`scripts/34`–`37`**).
6. **Low-rank SwiGLU MLP (gate / up / down)** on a single layer is **deployable on PPL** but **does not yet match teacher FFN direction** on captures. Best joint run (layer 8, rank 512/map, `both(w=0.25)`): **~0.69 capture cosine**, **~1.09× PPL**. Details below.

None of these replace a full LM benchmark harness; distributions are rarity-stratified Dolma windows on purpose.

---

## Low-rank SwiGLU MLP — what works, what does not, next steps

**Goal:** replace `gate_proj`, `up_proj`, and `down_proj` with rank-capped branches (supervised **output PCA** init, same recipe as Q/K), using **`ffn_input`** captures from **`scripts/27_capture_layer_internals.py`**.

**Code:** `src/attention_compression/mlp_lowrank.py`; **`scripts/48`** (joint), **`51`** (staged), **`49`** (PPL smoke), **`50`** (loss sweep). Full numbers: [`research/FINDINGS.md`](research/FINDINGS.md) § *Low-rank SwiGLU MLP*.

### What works

| Approach | Signal |
| --- | --- |
| **Joint** train all three maps (`48`, loss **`both(w=0.25)`**) | **~0.69** full-FFN cosine on layer-8 captures; **~1.09×** single-layer PPL (`49`) |
| **Supervised PCA init** on **pre-activation** gate/up linear outputs; down on `down_proj(h)` | Strong starting point; ranks hit cap 512 @ 95% variance on L8 |
| **Staged isolated** train (`51 --loss-target isolated`) | **~0.85–0.91** cosine *per linear map* on exact I/O |
| **MLP refresh after gate only** (`51 --mlp-refresh-epochs`, dense up/down still teacher) | Composed capture cosine **0.894 → 0.904** (320 windows/bin) |

### What does not (yet)

| Approach | Signal |
| --- | --- |
| **Block factorized FFN** (`43`–`45`, input-PCA / operator-Jacobian blocks) | **~0.50–0.55** capture cosine, **~1.14×** PPL — wrong inductive bias vs coupled SwiGLU |
| **Staged full-MLP loss** per stage (frozen earlier maps) | Gate **~0.91** alone, final composed **~0.69** — error compounds when up/down added |
| **Staged isolated + no refresh** | High per-map linear cosine but composed FFN **~0.62** (320/bin) |
| **MLP refresh after up/down** (all compressed maps, same LR) | **Unstable** — up refresh drops composed cosine; down refresh can blow up (**8×+** PPL) |

### Future directions

1. **Joint finetune** from staged/PCA init (`51 --finetune-epochs`) with full-MLP loss — close the gap between per-map linear fit and composed output.
2. **Gate-only MLP refresh** with **lower LR** (`--mlp-refresh-lr`); **skip refresh** when eval cosine regresses.
3. **Capture `mlp_hidden`** (`act(gate)·up`) in script 27 so down PCA/train uses the tensor `down_proj` actually sees.
4. **More rank / structured maps** only if PPL budget allows — 512/map is already at the PCA cap on L8.
5. Keep **Q/K + FFN** paths separate until a single **end-to-end** compressed block matches **`teacher_mlp(ffn_in)`** (bottleneck bridge lesson from script 32).

```bash
# Example: joint low-rank MLP, layer 8 (run on GPU host; PYTHONPATH=src)
python scripts/48_train_lowrank_mlp.py \
  --internals-capture-dir /path/to/layer08_internals_160pb \
  --output-dir /path/to/out --loss-kind both --loss-relative-weight 0.25

# Staged + isolated per-map I/O + optional refresh after each stage
python scripts/51_train_lowrank_mlp_staged.py \
  --internals-capture-dir /path/to/layer08_internals_320pb \
  --output-dir /path/to/out --loss-target isolated \
  --train-windows-per-bin 0 --mlp-refresh-epochs 3
```

---

## Progression of ideas (compact timeline)

This is narrative order—not script order—and matches the arcs recorded in **`research/FINDINGS.md`**.

| Phase | Rough focus | Representative code |
| --- | --- | --- |
| **Data plane** | Tokenize corpus, sparse bigram counts, rarity-scored sliding windows | `scripts/02`–`06` |
| **Geometry** | Plain PCA spectra; supervised PCA / regression sanity | `scripts/10`–`11` |
| **QKV drift vs teacher** | Empirical PCA on Q/K/V, replace in the forward → measure logits, attention KL, head context | `scripts/12` |
| **Trainable low-rank Q/K + dense V** | Joint/co-trained branches; asymmetric rank ladders | `scripts/13`, `scripts/16` |
| **Surgery @ scale** | Patch every layer × head Q/K paths; stratified perplexity deltas; prompts | `scripts/18`, `scripts/19` |
| **Stabilization** | Q/K-only LM fine-tune on selected windows after patch | `scripts/20` |
| **Speed** | Fused forwards, dense materialization experiments, prototype Triton | `scripts/21`, `scripts/26` |
| **Architectural forks** | “Sealed” compressed block (`22`), dense-smaller-score-path dead ends (`23`–`25`) | |
| **Black-box internals** | Capture **`head_context` + FFN_input** (`27`), bottleneck AEs (`28`–`30`), **`o_proj` mimic** (`31`), **post-mimic FFN** (`32`–`37`) → multi-layer pipelines (`33`–`35`) | |
| **Low-rank SwiGLU MLP** | Joint (`48`), staged/isolated (`51`), block-FFN dead end (`43`–`45`), operator SVD analysis (`45`) | |

Treat earlier sections in **`FINDINGS.md`** as the canonical numerics and remote paths (`/mnt/sdb1/dolma-v1_6-sample/` on your GPU host).

---

## Repository layout

```text
src/attention_compression/   importable Python package (metrics, loaders, PCA helpers, **qk_surgery**)
configs/                      small JSON defaults
scripts/                      numbered experiment entrypoints (`01`, `02`, … `51`; see FINDINGS script index too)
tests/                        pytest smoke / unit coverage where cheap
research/                      work logs—not the short story (see README there)
pyproject.toml               setuptools packaging from `src/`; extras for torch/transformers
```

Heavy artifacts (captures, checkpoints, sweep JSON under Terabytes-scale trees) stay **outside** git—see **`research/FINDINGS.md`** Remote Data State section.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tokenize,dev]"   # pull torch + transformers for model-facing scripts
pytest
```

Individual scripts assume you point them at corpus paths and GPU hosts per your infra; many runs in your notes used a remote Ubuntu workstation with **`/mnt/sdb1`** (see Cursor rules / [`research/FINDINGS.md`](research/FINDINGS.md)).

### CLI (`attention-compress`)

The CLI exposes four **targets** that match how you think about compression:

| `--target` | Trains | Applies |
| --- | --- | --- |
| `head` | Q/K (+ dense V) for **one** attention head (`--layer`, `--head`) | Q/K for the **whole layer** once every head checkpoint exists |
| `oproj` | Low-rank **`o_proj` swap** on one layer (script **31** checkpoint) | Replaces `attn.o_proj`; output dim unchanged (~75% params at rank 768) |
| `ffn` | FFN block on **one** layer: capture 27 → mimic `o_proj` 31 → bottleneck MLP 32 | Swaps script-31 `o_proj` + script-32 **MLP bridge** (teacher `mlp` kept); see FINDINGS |
| `layer` | All Q/K heads on the layer **plus** the FFN block | Both tracks on that layer |
| `model` | Repeat `layer` for each decoder layer (`--layers` to subset) | Same, per layer |

Dead-end tracks (sealed blocks, dense-small-QK, etc.) stay in `scripts/` and [`research/FINDINGS.md`](research/FINDINGS.md).

```bash
pip install -e ".[cli]"

# Readiness for a full-model compression plan
attention-compress plan allenai/OLMo-1B-0724-hf \
  --checkpoints /path/to/artifacts --target model

# One attention head (09 → 16)
attention-compress train allenai/OLMo-1B-0724-hf \
  --checkpoints /path/to/artifacts --target head --layer 0 --head 3 \
  --selected-csv /path/to/selected_train_windows.csv

# One layer's FFN block (27 → 31 → 32; needs frozen head-concat AE)
attention-compress train allenai/OLMo-1B-0724-hf \
  --checkpoints /path/to/artifacts --target ffn --layer 0 \
  --selected-csv /path/to/selected_train_windows.csv \
  --ae-state /path/to/head_concat_autoencoder.pt

# Whole layer, then whole model
attention-compress train ... --target layer --layer 0 --selected-csv ... --ae-state ...
attention-compress apply allenai/OLMo-1B-0724-hf \
  --checkpoints /path/to/artifacts --target model \
  --output /path/to/compressed-model --skip-missing --materialize-dense-qk
```

### Use in another project

After `pip install -e .`, import the surgery API from the installed package:

```python
from attention_compression.qk_surgery import (
    load_layer_qk_states,
    patch_layer_qk_dense_v,
    MultiHeadQKLowRankProjection,
)
```

Or add **`<this-repo>/src`** to **`PYTHONPATH`** and use the same imports without installing (equivalent to editable install for imports only).

---

## Contributing / extending

- Prefer **numbered scripts** for one-off pipelines; stash narrative + tables in **`research/`**.

- **Install:** `pip install -e .` from the repo root (package is resolved from **`src/`**). For ad-hoc runs without install, prepend **`<repo>/src`** to **`PYTHONPATH`** before `python scripts/...`.
- If you introduce a genuinely “default” artifact layout for a milestone, summarize it briefly in **`README.md`** and link to the exhaustive table in **`research/FINDINGS.md`**.
