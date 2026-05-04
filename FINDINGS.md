# Findings So Far

## Project Scope

This repo is implementing a distribution-aware local attention-compression experiment for `allenai/OLMo-1B-0724-hf` using the Dolma `v1_6-sample` corpus as a reproducible proxy for the model's data distribution.

The immediate goal is not full-model compression. The current milestone is to build and validate:

1. Dolma tokenization with the OLMo tokenizer.
2. Empirical unigram/bigram transition counts.
3. Window rarity scoring and rarity-bin selection.
4. Layer/head activation capture.
5. First compression diagnostics for one attention head.

## Remote Data State

Remote working data lives under:

`/mnt/sdb1/dolma-v1_6-sample`

Current key assets:

- Decompressed Dolma JSONL: `data/`
- Token shards: `tokens/`
- Sparse transition counts: `counts/`
- Scored windows: `windows/`
- Selected windows: `selected_windows/`
- Reports: `reports/`
- Captured activations: `activations/`
- PCA/RRR/drift artifacts: `pca/`, `rrr/`, `qkv_drift/`

Early in the project, free space after deleting retained `.json.gz` files was about `76G`; after the one-head 10k activation capture it was about `35G`. During later full-layer sweeps and experiments, usage shifted again — **always check `df` on `/mnt/sdb1` before large captures or training**.

## Corpus Format

The decompressed Dolma sample is newline-delimited JSON:

- Files: `103` `.json` files.
- Decompressed `.json` size: about `44.55 GiB`.
- Text field: `text`.
- Other observed fields: `id`, `source`, `metadata`, `added`, `created`, `version`.

The retained `.json.gz` files were deleted to recover space.

## Tokenization

Model/tokenizer:

`allenai/OLMo-1B-0724-hf`

Tokenizer/config facts:

- `tokenizer_len`: `50280`
- `tokenizer_vocab_size`: `50280`
- `config_vocab_size`: `50304`
- effective vocab size: `50304`
- EOS token: `<|endoftext|>`
- EOS token ID: `50279`
- token dtype: `uint16`

Tokenization output:

- Token shards: `161`
- Total tokens: `8,039,098,124`
- Last shard is smaller than the rest, as expected.

## Bigram Counts

Counts built from all token shards:

- Total tokens: `8,039,098,124`
- Total transitions: `8,039,097,963`
- Bigram run chunks: `321`
- Unique observed bigrams: `147,243,847`

Outputs:

- `counts/unigram_counts.npy`
- `counts/bigram_counts_sparse.npz`
- `counts/counts_summary.json`

Sparse key format:

```text
key = previous_token * vocab_size + next_token
```

## Window Scoring

Scored non-overlapping windows:

- `seq_len = 1024`
- `stride = 1024`
- Total scored windows: `7,850,613`

Output:

`windows/window_scores.npz`

Rarity score:

```text
mean_log_transition = mean_i log P(token_{i+1} | token_i)
rarity_score = -mean_log_transition
```

Rarity-bin distribution:

| Bin | Count | Mean Rarity |
| --- | ---: | ---: |
| very_common | 392,531 | 4.65 |
| common | 1,177,592 | 5.26 |
| typical | 4,710,367 | 5.63 |
| rare | 1,177,592 | 6.05 |
| very_rare | 314,024 | 6.50 |
| extreme_rare | 78,507 | 7.12 |

## Selected Windows

Selected train/eval manifests:

- `selected_windows/selected_train_windows.csv`
- `selected_windows/selected_eval_windows.csv`
- `selected_windows/selected_windows.npz`
- `selected_windows/selection_summary.json`

Selection:

- Train: `130,000` windows.
- Eval: `12,000` windows.
- Per-shard cap used: `500`.

Train bins:

| Bin | Train Count |
| --- | ---: |
| extreme_rare | 10,000 |
| very_rare | 20,000 |
| rare | 30,000 |
| typical | 40,000 |
| common | 20,000 |
| very_common | 10,000 |

Eval bins:

`2,000` windows per bin.

Validation:

- Loaded and checked `1000` train + `1000` eval selected windows.
- Validation errors: `0`.
- Train/eval selections touch all `161` token shards.

## Model Smoke Test

Model-facing smoke test passed:

- Loaded OLMo on CUDA.
- dtype: `bfloat16`.
- Transformer layers found at `model.layers`.
- Number of layers: `16`.
- Hidden size: `2048`.
- Attention heads: `16`.
- Head dim: `128`.
- Target layer used: `8`.

Layer-hook smoke shape:

```text
input:  [2, 1024, 2048] bf16 cuda:0
output: [2, 1024, 2048] bf16 cuda:0
```

## Activation Capture

### Full Layer Smoke Capture

Captured whole-layer input/output for `48` train windows:

- `8` windows per rarity bin.
- Layer: `8`.
- Shapes:
  - `x`: `[16, 1024, 2048]` per shard.
  - `y`: `[16, 1024, 2048]` per shard.
- Output size: about `385M`.

This proved the full-layer capture path works, but full-layer `x + y` is expensive.

### One-Head Capture

Switched to the more useful and storage-aware format:

- Store shared attention input once: `x_attn`.
- Store one head target: pre-`o_proj` `head_context`.

Captured:

- Layer: `8`
- Head: `0`
- Windows: `9,996`
- `1666` windows per rarity bin.
- Shards: `79`
- Output size: about `42G`

Per-shard tensor shapes:

```text
x_attn:       [128, 1024, 2048] bf16
head_context: [128, 1024, 128]  bf16
```

The target kind is:

```text
pre_o_proj_head_context
```

This is the current main dataset for first one-head surrogate experiments.

## Plain PCA Findings

### Head Context PCA

PCA target:

`head_context`, dimension `128`.

Sampled token positions: `1,290,240`.

Dimensions needed for explained variance:

| Explained Variance | Dims |
| --- | ---: |
| 50% | 37 |
| 75% | 75 |
| 80% | 84 |
| 90% | 105 |
| 95% | 116 |
| 99% | 126 |

Interpretation:

The final per-head context is not strongly low-rank under plain PCA. Variance is spread across most of the 128 dimensions.

### Shared Input PCA

PCA target:

`x_attn`, dimension `2048`.

Sampled token positions: `323,584`.

Dimensions needed:

| Explained Variance | Dims |
| --- | ---: |
| 50% | 382 |
| 75% | 968 |
| 80% | 1129 |
| 90% | 1515 |
| 95% | 1753 |
| 99% | 1980 |

Interpretation:

The layer input is also broadly distributed. A 640-dimensional input bottleneck would preserve only roughly the middle of the variance distribution, not nearly all of it.

## Q/K/V Empirical PCA

Computed empirical PCA for layer 8, head 0:

```text
Q = x_attn @ Wq_head
K = x_attn @ Wk_head
V = x_attn @ Wv_head
```

Dimensions needed:

| Tensor | 50% | 75% | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q | 14 | 39 | 48 | 73 | 91 | 113 |
| K | 7 | 26 | 35 | 61 | 82 | 108 |
| V | 38 | 76 | 85 | 105 | 116 | 126 |

Interpretation:

- `K` is most compressible by plain PCA.
- `Q` is moderately compressible.
- `V` is broad and looks much more like `head_context`.

This supports asymmetric treatment of routing (`Q/K`) versus payload/content (`V`).

## Supervised PCA / RRR Findings

First RRR baseline predicted `head_context_t` from token-local `x_attn_t`.

Training sample:

- `323,584` token positions.

Eval sample:

- `80,896` token positions.

Rank sweep:

| Rank | MSE | Relative MSE | Centered Cosine |
| ---: | ---: | ---: | ---: |
| 1 | 0.00763 | 0.940 | 0.245 |
| 2 | 0.00750 | 0.924 | 0.275 |
| 4 | 0.00731 | 0.901 | 0.315 |
| 8 | 0.00703 | 0.865 | 0.367 |
| 16 | 0.00663 | 0.816 | 0.428 |
| 32 | 0.00614 | 0.757 | 0.493 |
| 64 | 0.00560 | 0.690 | 0.557 |
| 96 | 0.00527 | 0.650 | 0.592 |
| 128 | 0.00505 | 0.622 | 0.615 |

Interpretation:

This token-local linear map is a weak baseline for predicting `head_context`, because the true head context depends on sequence-wide attention. It is useful as a lower bound, not as a final compression strategy.

## Q/K/V PCA Replacement Drift

The more relevant experiment:

1. Fit empirical PCA bases for `Q`, `K`, and `V`.
2. Reconstruct `Q_hat`, `K_hat`, `V_hat` at asymmetric ranks.
3. Apply OLMo RoPE to `Q/K`.
4. Recompute causal attention logits and probabilities.
5. Recompute `head_context_hat = A_hat @ V_hat`.
6. Compare to teacher `head_context`.

Small sweep setup:

- Train fit: `32` windows per rarity bin.
- Eval: `8` windows per rarity bin.
- Layer: `8`
- Head: `0`
- RoPE applied.
- Logit MSE computed only over valid causal positions.

Global results:

| Config | Q rel MSE | K rel MSE | V rel MSE | Logit rel MSE | Attn KL | Top1 | Top5 | Head rel MSE | Head Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q64 K64 V128 | 0.0327 | 0.0201 | ~0 | 0.0418 | 0.0371 | 0.746 | 0.945 | 0.0910 | 0.975 |
| Q96 K64 V128 | 0.0097 | 0.0201 | ~0 | 0.0276 | 0.0295 | 0.759 | 0.952 | 0.0687 | 0.981 |
| Q64 K48 V128 | 0.0327 | 0.0315 | ~0 | 0.0478 | 0.0404 | 0.734 | 0.941 | 0.0996 | 0.973 |
| Q96 K64 V96 | 0.0097 | 0.0201 | 0.138 | 0.0276 | 0.0295 | 0.759 | 0.952 | 0.600 | 0.515 |
| Q128 K96 V96 | ~0 | 0.0054 | 0.138 | 0.0071 | 0.0102 | 0.877 | 0.968 | 0.563 | 0.501 |
| Q64 K64 V96 | 0.0327 | 0.0201 | 0.138 | 0.0418 | 0.0371 | 0.746 | 0.945 | 0.617 | 0.514 |

Interpretation:

With `V` kept full-rank, compressed `Q/K` preserve head output well:

- `Q96 K64 V128`: head relative MSE about `0.069`, cosine about `0.981`.
- `Q64 K64 V128`: head relative MSE about `0.091`, cosine about `0.975`.
- `Q64 K48 V128`: head relative MSE about `0.100`, cosine about `0.973`.

When `V` is compressed to `96`, quality drops sharply:

- Head relative MSE rises to about `0.56-0.62`.
- Head cosine drops to about `0.50-0.52`.

Main conclusion:

```text
Q/K routing is compressible.
V/content is the hard part.
```

The most promising first candidate is:

```text
Q=96, K=64, V=128
```

A cheaper candidate worth testing further:

```text
Q=64, K=48, V=128
```

## Current Scientific Read

The strongest finding so far is not that the whole head is low-rank. It is that routing and payload behave differently:

- `Q/K` activations concentrate in lower-dimensional empirical subspaces.
- Attention-logit and attention-map metrics remain strong after Q/K PCA compression.
- `V` and final head context remain broad and are much more sensitive to compression.

This supports an asymmetric attention-compression story:

```text
routing channels (Q/K): more compressible
payload/content channels (V): less compressible, need different objective or more budget
```

It also supports measuring compression through attention behavior rather than through separate Q/K reconstruction alone.

## What Co-Training Means for Dimensionality

The co-trained Q/K/V branch experiment does not mean that PCA-discarded dimensions were simply restored for free.

The trainable branch has the form:

```text
Q_hat = X down_q up_q + bias_q
K_hat = X down_k up_k + bias_k
V_hat = X down_v up_v + bias_v
```

For a branch with rank `r`:

```text
down: [2048, r]
up:   [r, 128]
```

so the effective map has rank at most `r`.

Therefore:

- A `Q64` branch cannot recover all 128 token-varying Q output dimensions.
- It can only produce Q values in a 64-dimensional affine output subspace.
- Co-training can rotate that 64-dimensional subspace away from the initial top-64 PCA directions.
- This means it may recover some low-variance PCA directions by giving up some high-variance PCA directions.

That is not cheating. It is exactly the advantage of using PCA as a prior rather than as the final objective: low-variance directions may matter more for attention behavior than their variance alone suggests.

What still needs to be measured:

- principal-angle/subspace overlap between initial PCA bases and learned branch output subspaces;
- Q/K/V norm ratios after training;
- logit standard-deviation ratios;
- larger held-out eval and rarity-bin breakdowns;
- whether the improvement persists across heads and layers.

## Realistic Compression Estimate

For the current tested case:

```text
model: allenai/OLMo-1B-0724-hf
layer: 8
head:  0
```

the current evidence supports substantial Q/K compression, but not V compression.

Current practical estimate:

```text
Q: 128 -> 64 or 96
K: 128 -> 48 or 64
V: 128 -> 128 for now
```

Candidate budgets:

| Config | Total QKV Dims | Reduction | Current Read |
| --- | ---: | ---: | --- |
| Q96 K64 V128 | 288 vs 384 | 25.0% | safest tested routing-compressed candidate |
| Q64 K64 V128 | 256 vs 384 | 33.3% | plausible |
| Q64 K48 V128 | 240 vs 384 | 37.5% | aggressive but promising |

The strongest current statement is:

> For layer 8 head 0, Q/K can likely be compressed by roughly 2x while preserving head behavior, provided V remains full-rank and the branches are co-trained under attention-output loss.

The statement that should not be made yet:

> The whole model can be compressed by 37.5%.

That requires repeating the experiment across more heads/layers and testing actual module replacement.

## Routing-Only Compression Framing

The `Q64 K48 V128` result should be described carefully.

Because `V=128` equals the full per-head value dimension, this configuration is not compressing V. It is best described as:

```text
Q/K-compressed, V-preserving
```

or:

```text
routing-compressed head
```

The key question answered by this baseline is:

> Can we compress Q/K heavily while keeping V intact?

For layer 8 head 0, the current answer appears to be yes.

The PCA-initialized `Q64 K48 V128` baseline already has:

| Metric | Value |
| --- | ---: |
| head_context_relative_mse | 0.09097 |
| head_context_cosine | 0.97484 |
| attention_KL | 0.03693 |
| attention_top5_overlap | 0.9412 |
| logit_relative_mse | 0.04652 |

This is a strong local routing-compression anchor result.

The immediate interpretation:

```text
A large fraction of this head's routing behavior survives with:
    Q: 128 -> 64
    K: 128 -> 48
    V: 128 -> 128
```

The clean scientific phrasing is:

> Attention routing subspaces appear compressible earlier than payload/value subspaces.

## What To Look For In Co-Training

Since the PCA baseline is already strong, co-training may not dramatically improve every metric. The expected positive pattern is:

```text
Q/K/V reconstruction:
    may worsen slightly

logit_relative_mse:
    should improve or remain stable

attention_KL:
    should improve

top-k attention overlap:
    should improve or remain stable

head_context_relative_mse:
    should improve

head_context_cosine:
    should improve or remain stable
```

The desirable result is not necessarily better independent Q/K/V reconstruction. It is:

```text
co-trained branches reconstruct Q/K/V slightly worse,
but preserve S, A, and H better.
```

That would directly support the claim that PCA is a useful prior, while coherent attention behavior should be the training objective.

## Compression-Ratio Accounting

For one head, with:

```text
hidden_dim = 2048
head_dim   = 128
```

the original conceptual per-head Q/K/V projection parameters are:

```text
3 * 2048 * 128 = 786,432
```

For a separate-branch low-rank factorization:

```text
Wq_head ≈ Pq Rq, where Pq: 2048 x rq, Rq: rq x 128
Wk_head ≈ Pk Rk, where Pk: 2048 x rk, Rk: rk x 128
Wv_head ≈ Pv Rv, where Pv: 2048 x rv, Rv: rv x 128
```

Parameter counts:

| Config | Parameters | Fraction of Original | Reduction |
| --- | ---: | ---: | ---: |
| Q96 K64 V128 | 626,688 | 79.7% | 20.3% |
| Q64 K64 V128 | 557,056 | 70.8% | 29.2% |
| Q64 K48 V128 | 522,240 | 66.4% | 33.6% |
| Q64 K48 V96 | 452,608 | 57.6% | 42.4% |
| Q64 K48 V64 | 382,976 | 48.7% | 51.3% |

These are local per-head Q/K/V projection counts, not full-model compression ratios.

The current best validated point is routing-only:

```text
Q64 K48 V128
```

which is about a `33.6%` local per-head Q/K/V parameter reduction while preserving V.

The next decisive question is whether:

```text
Q64 K48 V96
```

or:

```text
Q64 K48 V64
```

can preserve enough head behavior after co-training.

## Next Asymmetric Compression Ladder

The next sweep should distinguish routing loss from payload loss.

Suggested configs:

```text
A. Q64  K48  V128
B. Q64  K48  V96
C. Q64  K48  V64
D. Q48  K32  V128
E. Q48  K32  V96
F. Q32  K24  V128
G. Q32  K24  V96
```

Add explicit ablations:

```text
Routing only:
    Q64  K48  V128

Payload only:
    Q128 K128 V96
    Q128 K128 V64

Both:
    Q64  K48  V96
    Q64  K48  V64
```

This will answer whether head error is dominated by routing compression or payload compression.

Current prediction:

```text
Q64 K48 V128:
    good

Q128 K128 V64:
    moderate degradation

Q64 K48 V64:
    much worse unless co-training helps substantially
```

## Rarity-Bin Reporting Requirement

The global metrics are encouraging, but the distribution-aware claim requires rarity-bin reporting.

For every config, report at least:

```text
head_context_relative_mse
attention_KL
top5_overlap
logit_relative_mse
```

for:

```text
very_common
common
typical
rare
very_rare
extreme_rare
```

The first important table should be:

```text
PCA init vs co-trained Q64 K48 V128 by rarity bin
```

If rare bins remain good, that is strong evidence that routing compression is robust. If rare bins are worse but co-training reduces the gap, that may be even more interesting for the paper.

## Joint Q/K/V Co-Training Ladder

After the first tiny co-training run, a larger ladder was launched using:

```text
train: 1024 windows per rarity bin = 6144 windows
eval:  256 windows per rarity bin = 1536 windows
epochs: 5
batch size: 2
learning rate: 5e-5
layer: 8
head: 0
```

The baseline `Q64 K48 V128` large run completed before the ladder:

| Config | PCA Init MSE | Final MSE | Final Cosine | Final Attn KL | Final Top5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q64 K48 V128 | 0.09097 | 0.01414 | 0.9949 | 0.00575 | 0.9712 |

This was a very strong result: co-training reduced head-context relative MSE from about `0.091` to about `0.014`.

The ladder results so far:

| Config | Role | PCA Init MSE | Final MSE | Final Cosine | Final Attn KL | Final Top5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q64 K48 V96 | routing + V96 | 0.6236 | 0.1519 | 0.9338 | 0.00569 | 0.9712 |
| Q64 K48 V64 | routing + V64 | 0.8122 | 0.3244 | 0.8505 | 0.00568 | 0.9712 |
| Q48 K32 V128 | aggressive routing, full V | 0.2351 | 0.0183 | 0.9942 | 0.00871 | 0.9645 |
| Q48 K32 V96 | aggressive routing + V96 | 0.7341 | 0.1571 | 0.9320 | 0.00869 | 0.9645 |
| Q32 K24 V128 | very aggressive routing, full V | 0.7766 | 0.0234 | 0.9926 | 0.01157 | 0.9592 |
| Q32 K24 V96 | very aggressive routing + V96 | 1.1417 | 0.1611 | 0.9308 | 0.01157 | 0.9592 |
| Q128 K128 V96 | payload-only V96 | 0.5511 | 0.1429 | 0.9369 | 0.00042 | 0.9935 |

The final payload-only `Q128 K128 V64` run was still in progress when this section was written. Interim values through epoch 2 were:

| Config | Stage | Head MSE | Cosine | Attn KL | Top5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Q128 K128 V64 | PCA init | 0.7552 | 0.3470 | ~0 | ~1.000 |
| Q128 K128 V64 | epoch 2 | 0.3253 | 0.8507 | 0.00113 | 0.9910 |

### Ladder Interpretation

The pattern is now very clear:

```text
V rank dominates head-output quality.
Q/K can be compressed very aggressively when V is preserved.
```

Evidence:

- `Q48 K32 V128` reaches `0.0183` head MSE and `0.9942` cosine.
- `Q32 K24 V128` reaches `0.0234` head MSE and `0.9926` cosine.
- These are only slightly worse than `Q64 K48 V128`.
- But any `V96` config clusters around `0.14-0.16` head MSE.
- `V64` is much worse, around `0.32` head MSE in both routing-compressed and payload-only settings.

This strongly suggests:

```text
Q/K routing can be much smaller than expected.
V/content is the primary bottleneck.
```

For this head, the surprising result is that:

```text
Q32 K24 V128
```

is already very good after co-training. That is:

```text
Q: 128 -> 32
K: 128 -> 24
V: 128 -> 128
```

Parameter count:

```text
Q: 2048*32 + 32*128 = 69,632
K: 2048*24 + 24*128 = 52,224
V: 2048*128 + 128*128 = 278,528
total = 400,384
```

Compared with original per-head Q/K/V:

```text
786,432 -> 400,384
```

This is about:

```text
50.9% of original local Q/K/V params
49.1% reduction
```

while preserving head output very well in this local experiment.

This is a stronger routing-compression result than initially expected.

### Current Best Local Point

For local head fidelity with full V:

```text
Q48 K32 V128
```

and:

```text
Q32 K24 V128
```

are both strong.

For more balanced compression including V:

```text
Q64 K48 V96
```

and:

```text
Q48 K32 V96
```

are viable but meaningfully worse.

The current practical conclusion:

> If V must be compressed, V96 is the first reasonable payload point. V64 is a large quality drop. If V can remain full-rank, Q/K can likely be compressed far more aggressively than originally expected.

## Direct Reduced-Head Activation Check

To separate real dimensional reduction from diagnostic zeroing/sparsification, I ran a direct dense-vs-low-rank activation check for a completed head:

```text
model: allenai/OLMo-1B-0724-hf
layer: 8
head: 13
eval windows: 6 (1 per rarity bin)
comparison: original dense Q/K/V head vs trained low-rank Q/K/V branch
```

The important distinction is:

```text
Q64 K48 V128 = Q/K dimension-reduced, V factorized but full rank for head_dim=128
Q64 K48 V96  = Q/K/V all actually dimension-reduced
```

For `Q64 K48 V128`, the reduced branch closely matched the dense head:

```text
head_context_relative_mse: 0.0158
head_context_cosine:       0.9944
attention_kl:              0.0078
attention_top5_overlap:    0.9339
q_relative_mse:            0.0662
k_relative_mse:            0.0659
v_relative_mse:            0.0038
```

For the actually dimension-reduced `Q64 K48 V96` branch:

```text
head_context_relative_mse: 0.1539
head_context_cosine:       0.9344
attention_kl:              0.0078
attention_top5_overlap:    0.9340
q_relative_mse:            0.0662
k_relative_mse:            0.0659
v_relative_mse:            0.1746
```

Per-head Q/K/V parameter count:

```text
original dense Q/K/V: 786,816
Q64 K48 V96:         452,992
reduction:           42.4%
```

Interpretation:

```text
Routing survives actual dimensional reduction well.
Payload quality is where V compression shows up.
```

The V96 run has essentially unchanged attention routing relative to V128, because Q/K are the same, but head-context relative MSE is about 10x worse:

```text
V128 head MSE: 0.0158
V96 head MSE:  0.1539
```

So the current result is not just sparse dense matrices or zeroed rows. The low-rank branch is genuinely implemented as skinny `down @ up` factors, and it can replace the original head activation path locally. The tradeoff is clear: real Q/K/V dimensional reduction buys meaningful parameter savings, but V rank below 128 causes a substantial payload degradation even when attention routing remains stable.

## Important Caveats

1. Many early head-level numbers are for one model, one layer, one head:

   `OLMo-1B-0724-hf`, layer `8`, head `0`.

   Separate sections above document **full-model** patching (16 layers × 16 heads) and remote timing; those still use **stratified Dolma windows**, not a standard LM benchmark suite (e.g. WikiText perplexity).

2. The 10k activation capture is rarity-stratified but still finite. Dimensionality estimates may change with more windows, other heads, other layers, or rare-only captures.

3. The Q/K/V drift sweep is currently small:

   `32` train windows per bin and `8` eval windows per bin.

4. Full validation should repeat the drift sweep with larger train/eval splits and compare by rarity bin.

5. **Inference timings** depend on PyTorch build, GPU, batch size, sequence length, and whether **fused Q/K** / **`torch.compile`** are enabled; reported numbers are from one remote GPU configuration.

## Suggested Next Steps

1. Run the Q/K/V PCA replacement-drift experiment at larger scale.
2. Repeat for several heads in layer 8.
3. Compare spectra/drift for:
   - random windows
   - rarity-stratified windows
   - rare-only windows
4. Implement V-specific objectives:
   - preserve `A @ V`
   - output-sensitive V compression through `W_o`
   - low-rank residual correction for V/head context
5. Start a trainable surrogate using the strongest baseline candidate:

```text
Q=96, K=64, V=128
```

and compare against:

```text
Q=64, K=48, V=128
```

## Expedited ladder: Q/K low-rank + dense V (all heads per layer)

After the full joint Q/K/V ladder on layer 8, subsequent layers used the faster protocol: train **only** low-rank Q/K branches with **V dense** (`scripts/16_train_qk_dense_v.py`, ranks **Q64 K48**).

Checkpoint layout under the remote tree:

```text
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/layer_XX_head_YY_q64_k48_densev/
    qk_dense_v_model.pt
    qk_dense_v_report.json
```

Layer **8** did not have this expedited sweep initially; for **full-model** wiring below, layer **8** used existing **`joint_qkv`** checkpoints (`q64_k48_v128`) only as a source for **Q/K branch tensors** (same low-rank factor shapes); **V stays dense** everywhere in this deployment path.

Layers covered by the expedited sweep (remote): **0–7, 9–15** (full **16 heads** each), plus layer **8** via fallback as above.

## Full-model surgery: patched Q/K + dense V

All **16** layers were patched: replace each layer’s **`q_proj`** and **`k_proj`** with per-head low-rank factors; **`v_proj`** remains the original dense linear.

### Next-token loss / perplexity (selected eval windows)

These are **not** benchmark-quality LM evals; they are distribution-stratified window loss on the **same** `selected_eval_windows.csv` setup used elsewhere.

**Small smoke (1 eval window per rarity bin, 6 windows, seq_len 1024)**

```text
baseline loss: 2.208   perplexity: ~9.10
patched loss:  2.275   perplexity: ~9.73
perplexity ratio: ~1.069
```

**Medium (32 eval windows per bin, 192 windows, ~196k tokens)**

```text
baseline loss: 2.680   perplexity: ~14.58
patched loss:  2.722   perplexity: ~15.21
perplexity ratio: ~1.043
```

**Larger (96 eval windows per bin, 576 windows, ~589k tokens)**

```text
baseline loss: 2.596   perplexity: ~13.41
patched loss:  2.641   perplexity: ~14.03
perplexity ratio: ~1.046   (~+4.6% perplexity vs baseline)
```

**Parameter accounting on the patched Q/K/V projection path only** (16 layers × 16 heads, Q64 K48 + dense V per head):

```text
patched_dense_qkv (hypothetical all-dense): 201,424,896
patched_qk_low_rank_dense_v (actual):       129,597,440
reduction on those projections:             ~35.66%
```

### Side-by-side generations

`scripts/19_qk_dense_v_compare_outputs.py` compares **baseline** vs **patched** on fixed prompts (greedy + sampled) and optional window-level loss. Qualitatively: translations and short completions often stay aligned; **greedy** paths can still **loop** on some prompts (e.g. “entropy”) even when average perplexity is only moderately worse.

Remote example JSON:

```text
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/compare_outputs.json
```

## Light fine-tune of the patched model (train split, Q/K only)

`scripts/20_qk_dense_v_finetune.py` loads the **already patched** model, freezes everything except the low-rank **Q/K** parameters, and runs a short **causal LM** objective on `selected_train_windows.csv`.

**First attempt (too aggressive — eval got worse)**

```text
train: 64 windows / bin (384 windows), eval: 32 / bin (192 windows)
steps: 400, lr: 5e-5, batch: 2
pre fine-tune eval loss:  2.722  perplexity: ~15.21
post fine-tune eval loss: 2.781  perplexity: ~16.14
```

**Second attempt (conservative — eval improved monotonically)**

```text
train: 192 windows / bin (1152 windows), eval: 32 / bin (192 windows)
steps: 300, lr: 1e-5, weight_decay: 0.01, batch: 2
pre fine-tune eval loss:  2.722  perplexity: ~15.21
post fine-tune eval loss: 2.710  perplexity: ~15.03
```

**Three-way comparison on the larger 576-window eval** (same protocol as above):

| Variant | Eval loss | Eval perplexity | vs baseline perplexity |
| --- | ---: | ---: | --- |
| Baseline | 2.596 | 13.409 | — |
| Patched (no fine-tune) | 2.641 | 14.032 | **+4.64%** |
| Patched + fine-tune v2 | 2.628 | 13.844 | **+3.24%** |

Fine-tuned checkpoints are written in the **same directory layout** as the original `qk_dense_v` tree for drop-in use with the surgery / compare scripts.

Remote example:

```text
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v_finetuned_v2/
    finetune_summary.json
    compare_outputs_finetuned.json
```

## Dtype at inference (baseline vs patched)

For CUDA runs, the **reference Hugging Face model** is loaded in **`bfloat16`**. The **patched** low-rank Q/K modules are constructed in **`bfloat16`** for inference so forward math matches the rest of the stack.

During **fine-tuning**, only the trainable Q/K low-rank tensors may be held in **`float32`** for optimizer stability; the module forward casts activations to the weight dtype and casts outputs back so the surround model stays bf16.

Checkpoint files on disk (`qk_dense_v_model.pt`, etc.) are often saved in **`float32`**; they are cast to bf16 when instantiated inside the live model.

## Inference speed (patched vs dense Q/K)

Naïve implementation (Python loop over heads: many small GEMMs) was **much slower** than the original single fused `Linear` per projection, even though parameter counts drop.

### Fused Q/K forward (`scripts/_qk_surgery_lib.py`)

**`MultiHeadQKLowRankProjection`** now uses a **fused** path when all heads share the same rank (the usual **Q64** or **K48** case):

1. `torch.cat` all `down` matrices along the rank dimension → one large matmul `hidden @ down_cat`.
2. Reshape to `[batch, seq, num_heads, rank]`.
3. `torch.einsum('bshr,hrd->bshd', ...)` for all head `up` matrices.
4. Add stacked biases and flatten head outputs.

If ranks differ per head, it **falls back** to the original loop. **`_force_naive_forward`** still forces the loop for debugging.

### Dense Q/K materialization (inference parity path)

At load time, **`materialize_dense_linear_from_branch_states`** folds each head’s `down @ up` into one **`nn.Linear`** per Q and per K (same single fused GEMM shape as vanilla HF projections). Checkpoints on disk stay **low-rank**; VRAM for live Q/K weights matches **dense** width during inference.

Use **`scripts/21_inference_speed_benchmark.py --materialize-dense-qk`**. When some layers lack `qk_dense_v` trees (this Dolma sample omits **layer 8**), pass **`--fallback-joint-qkv-root`** like the fused runs.

### Benchmarks (`scripts/21_inference_speed_benchmark.py`)

Setup (remote, **OLMo-1B-0724-hf**, bf16, batch 1): **prefill** one forward on **[1, 1024]** with `use_cache=False`; **decode** time only the loop of **128** single-token steps with KV cache (after untimed prefill on **512** tokens). Median over **15** repeats after **5** warmups. Patched rows below that use fallback include **`--fallback-joint-qkv-root /mnt/sdb1/dolma-v1_6-sample/joint_qkv`** (layer 8).

| Configuration | Prefill (ms / forward) | Decode (ms / token) | vs baseline |
| --- | ---: | ---: | --- |
| Baseline dense | ~15.34 | ~8.18 | 1.00× |
| Patched, **naive** per-head loop | ~31.89 | ~26.50 | ~2.05× / ~3.19× slower |
| Patched, **fused** default | ~16.60 | ~14.44 | ~1.08× / ~1.76× slower |
| Patched, fused + **`torch.compile`** | ~14.97 | ~14.95 | ~0.97× / ~1.81× |
| Patched, **materialized dense Q/K** (`--materialize-dense-qk`) | ~15.66 | ~8.62 | ~1.01× / ~1.06× |

**`torch.compile` notes:** Hugging Face causal LM + KV cache interacts badly with Inductor **CUDAGraph trees** on this stack. The benchmark disables them by default:

```python
torch._inductor.config.triton.cudagraph_trees = False
torch._inductor.config.triton.cudagraphs = False
```

The script also calls **`torch.compiler.cudagraph_mark_step_begin()`** before forwards when compiling (harmless when graphs are off). **`--compile-enable-cudagraph`** re-enables trees for experiments.

Remote JSON examples:

```text
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/inference_speed_fused.json
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/inference_speed_naive_loop.json
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/inference_speed_fused_compile.json
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/inference_speed_materialized_dense_qk.json
```

**Read:** fusion restores **prefill** to rough parity but **decode** stayed slower because Q/K still avoided one fat GEMM per projection. **Materializing** Q/K to dense **`Linear`** restores **decode** to baseline-class throughput (~within a few percent on this run); trade-off is **full Q/K weight VRAM** while checkpoints remain low-rank on disk.

### Triton fused low-rank kernel prototype

Added **`scripts/26_triton_lowrank_qk_benchmark.py`** to test a custom fused kernel for the working low-rank math:

```text
out_h = x @ down_h @ up_h + bias_h
```

The Triton prototype launches one program per head/tile, computes `x @ down_h` into registers, immediately applies `up_h`, and writes the full 128-dim head output. It does **not** materialize dense Q/K weights. This is synthetic projection-only timing, not yet integrated into the Hugging Face model.

Remote synthetic benchmark, bf16, OLMo shape **D=2048, H=16, head_dim=128**:

| Shape | Best Triton setting | Dense materialized | Current PyTorch packed low-rank | Triton fused low-rank | Read |
| --- | --- | ---: | ---: | ---: | --- |
| Q rank 64, decode-like `tokens=1` | `BLOCK_M=4/8/16` | ~0.037 ms | ~0.079 ms | ~0.028 ms | Triton is ~0.75× dense and ~0.35× PyTorch low-rank |
| K rank 48, decode-like `tokens=1` | `BLOCK_M=16` | ~0.038 ms | ~0.079 ms | ~0.029 ms | Similar decode win |
| Q rank 64, prefill-like `tokens=1024` | `BLOCK_M=128` | ~0.059 ms | ~0.104 ms | ~0.046 ms | Triton can beat dense in synthetic prefill when tiled larger |
| K rank 48, prefill-like `tokens=1024` | `BLOCK_M=16` | ~0.059 ms | ~0.104 ms | ~0.079 ms | Faster than PyTorch low-rank, slower than dense in this setting |

Read: this is the first path that appears to satisfy all three constraints in a projection-only benchmark: **keep low-rank live weights**, **preserve full 128-dim Q/K geometry**, and **avoid the PyTorch low-rank decode penalty**. Next step is integration into `MultiHeadQKLowRankProjection`, likely with different tile choices for decode vs prefill and separate Q/K rank specializations.

## Sealed compressed layer prototype

Motivation: keep the model's residual/embedding width **D** unchanged at layer boundaries, but make each layer a hermetic compressed unit internally:

```text
D -> H*r -> attention in r-space -> D
D -> ffn_small -> D
```

This directly tests the hypothesis that each head uses only a limited, head-specific subspace of the residual stream while the full **D** residual space remains useful for inter-layer communication.

Added **`scripts/22_train_sealed_compressed_block.py`**. It trains a replacement block from **`scripts/08_capture_layer_activations.py`** shards (`x -> y`), with:

- one fused down-projection **`D -> H*r`** into per-head subspaces,
- per-head **`r x r`** Q/K cores and **`r x value_rank`** V cores,
- causal attention in compressed head space,
- explicit per-head **`value_rank -> D`** lifts, summed into the residual stream,
- smaller FFN **`D -> ffn_dim -> D`**,
- near-identity residual initialization so epoch 0 is close to the original layer input baseline.

Tiny remote smoke (layer 0, **12** windows total: **1 train + 1 eval per rarity bin**, seq len 1024). Metrics include both **relative MSE** on full layer output (`yhat` vs `y`) and **delta relative MSE** on the residual update (`yhat - x` vs `y - x`):

| Prototype | Params | Train / eval windows | vs OLMo layer params | Best eval relative MSE | Best eval delta relative MSE | Best eval cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r=32`, `value_rank=32`, `ffn_dim=512` | 4.25M | 6 / 6 | ~6.3% | ~0.685 | ~0.754 | ~0.554 |
| `r=64`, `value_rank=64`, `ffn_dim=1024` | 8.59M | 6 / 6 | ~12.8% | ~0.582 | ~0.640 | ~0.608 |
| `r=208`, `value_rank=208`, `ffn_dim=3584` | 30.39M | 36 / 12 | ~45.3% | ~0.531 | ~0.582 | ~0.663 |
| `r=208`, `value_rank=208`, `ffn_dim=3584` | 30.39M | 768 / 192 | ~45.3% | ~0.354 | ~0.390 | ~0.785 |
| `r=208`, `value_rank=208`, `ffn_dim=3584`, supervised PCA init (`q64/k48/v96`) | 30.40M | 768 / 192 | ~45.3% | ~0.362 | ~0.398 | ~0.778 |
| `r=208`, `value_rank=208`, `ffn_dim=3584`, explicit per-head lifts | 30.40M | 768 / 192 | ~45.3% | ~0.353 | ~0.389 | ~0.785 |
| `r=208`, `value_rank=208`, `ffn_dim=3584`, explicit per-head lifts + supervised PCA init (`q64/k48/v96`) | 30.40M | 768 / 192 | ~45.3% | ~0.360 | ~0.396 | ~0.780 |
| `r=208`, `value_rank=208`, `ffn_dim=3584`, explicit lifts + head contribution loss `0.05` | 30.40M | 768 / 192 | ~45.3% | ~0.354 | ~0.390 | ~0.785 |
| `r=208`, `value_rank=208`, `ffn_dim=3584`, explicit lifts + head contribution loss `0.25` | 30.40M | 768 / 192 | ~45.3% | ~0.362 | ~0.399 | ~0.780 |

Baseline OLMo layer 0 parameter count on this model: **67.1M** total (**16.8M attention**, **50.3M MLP**). These results are only a shape/training smoke: the split is too small to judge quality, and the architecture omits OLMo-specific details such as RoPE and gated MLP structure. The important signal is that the sealed design runs and learns from whole-layer targets while preserving the external **D -> D** interface.

The supervised PCA init in **`22`** follows the earlier branch recipe: compute teacher **Q/K/V output PCA** from captured training activations, set each branch slice as `down = W @ U`, `core = U.T`, and use `mean - mean @ U @ U.T` as a small output bias. For whole-layer captures, PCA is fit on the teacher **input-layernormed** activations before Q/K/V. In this first packed shared-subspace version, it did **not** beat near-identity random init; likely causes are that the sealed architecture shares one head subspace across Q/K/V slices, currently lacks RoPE in the compressed attention path, and only initializes the attention half while the FFN remains randomly trained.

The output projection is now represented explicitly as per-head lifts (`ctx_h @ lift_h`, then sum over heads). This is mathematically equivalent to slicing a concatenated `o_proj`, but makes the intended head-local output subspace explicit and gives a cleaner target for future per-head contribution losses.

Per-head contribution supervision was added by computing the teacher attention contribution **before** summing heads (`teacher_ctx_h @ o_proj_h`) and comparing it against each compressed head's lifted contribution. With weight **0.05**, output quality stays basically tied with the unconstrained explicit-lift run while head-contribution relative MSE improves to **~0.908** (from ~1.003 at init). With weight **0.25**, contribution MSE improves further to **~0.825**, but layer-output relative MSE worsens to **~0.362**. This suggests the contribution loss is valid but needs scheduling/annealing or a better decomposed target to avoid fighting the summed layer-output objective.

Remote artifacts:

```text
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_capture_tiny
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r32_ffn512_tiny_zeroish
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r64_ffn1024_tiny_zeroish
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_capture_8pb
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_8pb
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_capture_160pb
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_160pb
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_160pb_pca_norm_init
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_160pb_head_lift
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_160pb_head_lift_pca
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_160pb_head_lift_contrib005
/mnt/sdb1/dolma-v1_6-sample/sealed_block_layer00_r208_ffn3584_160pb_head_lift_contrib025
```

## Reduced dense Q/K dimension experiment

Motivation: avoid low-rank two-stage runtime shape entirely by making Q/K true smaller dense linears:

```text
q_h: D -> qk_dim
k_h: D -> qk_dim
v_h: D -> 128  # unchanged dense V
```

Added **`scripts/23_train_reduced_qk_dim.py`**. It tests the original prune recipe on one head:

1. fit supervised PCA over teacher Q/K outputs,
2. initialize dense **`D -> start_qk_dim`** Q/K projections,
3. train learnable gates with an L1 sparsity penalty,
4. fold gates into weights and prune to **`target_qk_dim`**,
5. fine-tune the pruned smaller dense projections against teacher logits/attention/head-context targets.

Remote smoke: **layer 0, head 0**, capture **64 windows/bin** (384 total), train/eval **288 / 96** windows.

| Variant | Q/K runtime shape | Eval logit rel MSE | Attention KL | Top-5 overlap | Head-context rel MSE | Head-context cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Existing Q64/K48 low-rank + dense V (`16`) | low-rank factors, full 128-dim Q/K output | ~0.0034 | ~0.0013 | ~0.985 | ~0.133 | ~0.997 |
| Reduced dense Q/K, gated before prune | dense `D -> 96` | ~0.277 | ~0.086 | ~0.867 | ~0.220 | ~0.964 |
| Reduced dense Q/K, prune `96 -> 64` + fine-tune | dense `D -> 64` | ~1.506 | ~0.160 | ~0.802 | ~0.340 | ~0.942 |

Read: the true smaller-dense Q/K shape is attractive for runtime, but this first PCA+gate+prune recipe is much worse than the existing low-rank Q/K branch on attention logits. The pruned model can recover head-context cosine reasonably, but logit fidelity remains poor. Two likely issues: arbitrary dimension deletion interacts badly with RoPE coordinate/frequency assignment, and Q/K PCA bases learned separately do not preserve the bilinear dot-product geometry as well as directly factorizing the original full Q/K maps.

Remote artifacts:

```text
/mnt/sdb1/dolma-v1_6-sample/reduced_qk_dim_layer00_head00_capture_64pb
/mnt/sdb1/dolma-v1_6-sample/reduced_qk_dim_layer00_head00_96to64
/mnt/sdb1/dolma-v1_6-sample/reduced_qk_dim_layer00_head00_96to64_gatefold
```

## Q64/K64 vs Q64/K48 low-rank branch

Motivation: the Triton low-rank kernel is much cleaner when rank is a power of two. K was originally **48** because PCA suggested fewer K dimensions were needed, but **64** may be a better engineering point if quality is at least as good.

Remote matched comparison: **layer 0, head 0**, same capture as reduced-dim tests (**288 / 96** train/eval windows), **5** epochs, dense V.

| Variant | Params vs dense QKV | Logit rel MSE | Attention KL | Top-5 overlap | Head-context rel MSE | Head-context cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q64/K48 + dense V | ~64.3% of dense QKV | ~0.00427 | ~0.00247 | ~0.978 | ~0.129 | ~0.996 |
| Q64/K64 + dense V | ~68.8% of dense QKV | ~0.00305 | ~0.00151 | ~0.982 | ~0.130 | ~0.997 |

Read: on this head, **K64 improves logits/KL/top-k** over K48 while keeping head-context quality essentially unchanged. It costs about **4.4 percentage points** of dense-QKV parameter fraction for this head, but makes Q/K ranks symmetric and power-of-two, which is much friendlier for Triton specialization.

Remote artifacts:

```text
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/layer_00_head_00_q64_k48_densev_64pb_compare
/mnt/sdb1/dolma-v1_6-sample/qk_dense_v/layer_00_head_00_q64_k64_densev_64pb_compare
```

## Shared lower-dimensional Q/K metric experiment

Motivation: instead of deleting Q/K dimensions independently, learn a shared score space:

```text
q_h: D -> q_rank -> shared_dim
k_h: D -> k_rank -> shared_dim
scores = q_shared @ k_shared.T
v_h: D -> 128  # unchanged dense V
```

Added **`scripts/24_train_shared_qk_metric.py`**. It initializes Q/K branches from supervised PCA, then learns auxiliary reshape matrices into a common `shared_dim`. This preserves the idea that Q and K must live in the same metric space while still avoiding full 128-dim Q/K logits.

Remote smoke: **layer 0, head 0**, same **288 / 96** train/eval windows as the reduced-dim experiment.

| Variant | Q/K runtime score dim | Eval logit rel MSE | Attention KL | Top-5 overlap | Head-context rel MSE | Head-context cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing Q64/K48 low-rank + dense V (`16`) | 128 full output | ~0.0034 | ~0.0013 | ~0.985 | ~0.133 | ~0.997 |
| Reduced dense Q/K, prune `96 -> 64` + fine-tune (`23`) | 64 | ~1.506 | ~0.160 | ~0.802 | ~0.340 | ~0.942 |
| Shared Q/K metric (`q64/k48 -> shared64`) (`24`) | 64 | ~0.345 | ~0.129 | ~0.829 | ~0.294 | ~0.955 |

Read: the learned shared metric is much better than hard-pruned 64-dim Q/K on logits, so the "auxiliary reshape into common score space" idea is directionally right. It is still far worse than the existing low-rank branch that reconstructs **full 128-dim Q/K** before RoPE/dot-product. Current evidence says full Q/K output geometry is carrying important information; compressing the score dimension itself is harder than compressing the projection parameters.

Remote artifact:

```text
/mnt/sdb1/dolma-v1_6-sample/shared_qk_metric_layer00_head00_q64_k48_s64
```

## Dense smaller Q/K trained directly on logits

Motivation: if inference must be **one GEMM smaller than dense**, the runtime shape has to be a true smaller dense projection:

```text
q_h: D -> qk_dim
k_h: D -> qk_dim
scores = q_h @ k_h.T
v_h: D -> 128  # unchanged dense V
```

Added **`scripts/25_train_dense_small_qk_logits.py`**. It trains **dense** `Wq_small` / `Wk_small` directly against teacher logits/KL/head-context, with shared-PCA initialization over concatenated teacher Q/K outputs. This is the closest implementation of the desired "single smaller GEMM" constraint.

Remote smoke: **layer 0, head 0**, same **288 / 96** train/eval windows.

| Variant | Q/K runtime score dim | Eval logit rel MSE | Attention KL | Top-5 overlap | Head-context rel MSE | Head-context cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing Q64/K48 low-rank + dense V (`16`) | 128 full output | ~0.0034 | ~0.0013 | ~0.985 | ~0.133 | ~0.997 |
| Dense small Q/K, shared-PCA init (`D -> 64`) (`25`) | 64 | ~0.379 | ~0.123 | ~0.831 | ~0.277 | ~0.959 |
| Dense small Q/K, shared-PCA init (`D -> 96`) (`25`) | 96 | ~0.151 | ~0.048 | ~0.901 | ~0.208 | ~0.976 |

Read: direct logit training of a true smaller dense Q/K projection is the best **single-GEMM** route so far, especially at **96** dims. It is still much worse than preserving full 128-dim Q/K geometry via low-rank reconstruction, but the gap narrows as the score dimension increases. This gives a concrete quality/speed trade-off axis: `qk_dim=96` saves 25% of Q/K output width/K-cache for the head while keeping one dense GEMM per Q/K.

Remote artifacts:

```text
/mnt/sdb1/dolma-v1_6-sample/dense_small_qk_layer00_head00_d64_sharedpca
/mnt/sdb1/dolma-v1_6-sample/dense_small_qk_layer00_head00_d96_sharedpca
```

## Black-box head PCA + FFN-input autoencoder

Motivation: inspect attention heads as black-box functions by looking at the PCA spectrum of each head's post-attention context output, and separately test whether the **full FFN input activation** can be compressed to half width.

Added:

- **`scripts/27_capture_layer_internals.py`** captures `head_contexts` at the input to `attn.o_proj`, reshaped as `[batch, seq, num_heads, head_dim]`, plus `ffn_input` at the input to the layer MLP.
- **`scripts/28_head_pca_ffn_autoencoder.py`** computes per-head PCA spectra and trains a linear `2048 -> 1024 -> 2048` autoencoder on FFN inputs.

Remote run: **layer 0**, **384** windows (**64 per rarity bin**), train/eval split **288 / 96** windows for analysis.

Head-context PCA ranks across the 16 heads:

| Explained variance | Min rank | Median rank | Mean rank | Max rank |
| ---: | ---: | ---: | ---: | ---: |
| 90% | 48 | 100.5 | 95.1 | 112 |
| 95% | 72 | 114.0 | 109.2 | 120 |
| 99% | 109 | 125.5 | 123.7 | 127 |

Half-width linear autoencoders:

| Target | Bottleneck | Params | Eval relative MSE | Eval cosine |
| --- | ---: | ---: | ---: | ---: |
| FFN input | `2048 -> 1024 -> 2048` | ~4.20M | ~0.0946 | ~0.951 |
| Concatenated head output before `o_proj` | `2048 -> 1024 -> 2048` | ~4.20M | ~0.1623 | ~0.924 |
| Independent per-head outputs | `16 x (128 -> 64 -> 128)` | ~0.26M | ~0.3504 | ~0.856 |

Deeper GELU MLP autoencoders, same bottleneck sizes:

| Target | Architecture | Eval relative MSE | Eval cosine |
| --- | --- | ---: | ---: |
| Concatenated head output before `o_proj` | `2048 -> 1536 -> 1024 -> 1536 -> 2048` | ~0.3477 | ~0.839 |
| Independent per-head outputs | `16 x (128 -> 96 -> 64 -> 96 -> 128)` | ~0.7790 | ~0.542 |

Linear-initialized nonlinear residual decoders:

| Target | Architecture | Eval relative MSE | Eval cosine |
| --- | --- | ---: | ---: |
| Concatenated head output before `o_proj` | linear `2048 -> 1024 -> 2048` + zero-init `1024 -> 1536 -> 2048` decoder residual | ~0.1487 | ~0.931 |
| Independent per-head outputs | linear `16 x (128 -> 64 -> 128)` + zero-init `64 -> 96 -> 128` decoder residuals | ~0.3251 | ~0.869 |

Training through frozen `o_proj`:

| Model | Pre-`o_proj` rel MSE | Pre-`o_proj` cosine | Post-`o_proj` rel MSE | Post-`o_proj` cosine |
| --- | ---: | ---: | ---: | ---: |
| Linear combined-head AE, activation-trained | ~0.1623 | ~0.924 | ~0.0920 | ~0.958 |
| Residual decoder combined-head AE, activation-trained | ~0.1487 | ~0.931 | ~0.0822 | ~0.962 |
| Residual decoder combined-head AE, **`o_proj`-trained** | ~0.1644 | ~0.923 | ~0.0765 | ~0.964 |

Distilling the autoencoder target into a replacement `o_proj`:

Target is **`teacher_o_proj(AE(x))`**, where `x` is the original concatenated head output. This trains a new projection from the original `2048`-d head output to residual space. The goal is to absorb the autoencoder correction into `o_proj`, optionally with low-rank parameters.

| Replacement `o_proj` | Trainable params vs dense `o_proj` | Rel MSE to `o_proj(AE(x))` | Cosine to `o_proj(AE(x))` | Rel MSE to teacher `o_proj(x)` | Cosine to teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense `2048 -> 2048` | ~100% | ~0.0012 best / ~0.0046 final | ~0.999 best / ~0.998 final | ~0.083-0.087 | ~0.960 |
| Low-rank rank 512 | ~50% | ~0.1019 | ~0.955 | ~0.1750 | ~0.919 |
| Low-rank rank 768 | ~75% | ~0.0414 | ~0.982 | ~0.1200 | ~0.944 |

Larger-data follow-up: captured **160 windows/bin** (960 windows) and trained/evaluated low-rank `o_proj` replacements with **128 / 32 windows/bin**. Same AE target as above.

| Replacement `o_proj` | Train/eval windows | Rel MSE to `o_proj(AE(x))` | Cosine to `o_proj(AE(x))` | Rel MSE to teacher `o_proj(x)` | Cosine to teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low-rank rank 512 | 768 / 192 | ~0.0974 | ~0.956 | ~0.1672 | ~0.922 |
| Low-rank rank 768 | 768 / 192 | ~0.0393 | ~0.982 | ~0.1141 | ~0.947 |

Supervised-PCA low-rank initialization: fit PCA on the target residual activation **`o_proj(AE(x))`**, set `up` to the top residual-space PCs, and initialize `down` by ridge regression from original head output `x` into the PCA coefficients. Same **160pb** capture and **128 / 32 windows/bin** split.

| Replacement `o_proj` | Init | Rel MSE to `o_proj(AE(x))` | Cosine to `o_proj(AE(x))` | Rel MSE to teacher `o_proj(x)` | Cosine to teacher |
| --- | --- | ---: | ---: | ---: | ---: |
| Rank 512 | supervised PCA + ridge | ~0.0966 | ~0.957 | ~0.1663 | ~0.923 |
| Rank 768 | supervised PCA + ridge | ~0.0385 best / ~0.0389 final | ~0.983 | ~0.1134 best / ~0.1137 final | ~0.947 |

Per-head half-width autoencoder detail:

| Head group | Heads by eval relative MSE |
| --- | --- |
| Best reconstructed | `3` (0.111, 0.948 cos), `7` (0.271, 0.882), `5` (0.286, 0.863), `9` (0.287, 0.853), `11` (0.320, 0.852) |
| Worst reconstructed | `0` (0.458, 0.780), `1` (0.454, 0.794), `13` (0.454, 0.790), `15` (0.423, 0.831), `10` (0.412, 0.831) |

Follow-up: raw head-coordinate PCA bases are not directly comparable across heads because each head has a private 128-d coordinate system. Added **`scripts/29_head_subspace_overlap.py`**, which maps each head's PCA basis through that head's `o_proj` slice and compares the resulting subspaces in the shared residual space. Overlap is normalized projection overlap (`1.0` = identical subspace).

Top shared residual-space subspaces:

| PCA rank | Median pair overlap | Mean pair overlap | Max pair overlap | Strongest pairs |
| ---: | ---: | ---: | ---: | --- |
| 32 | 0.0176 | 0.0223 | 0.1040 | `(3,7)`, `(4,7)`, `(3,4)`, `(3,8)` |
| 64 | 0.0183 | 0.0242 | 0.1273 | `(3,8)`, `(3,7)`, `(3,4)`, `(4,7)` |
| 96 | 0.0223 | 0.0301 | 0.1787 | `(3,8)`, `(3,7)`, `(3,9)`, `(3,4)` |
| 112 | 0.0258 | 0.0339 | 0.1964 | `(3,8)`, `(3,7)`, `(3,9)`, `(3,4)` |

Read: the head context outputs are not strongly low-dimensional if the target is high variance preservation; most heads need close to the full 128 dimensions for 95-99% variance, though a few heads are more compressible. Across heads, most residual-space PCA subspaces are close to orthogonal/weakly overlapping, but **heads 3, 7, and 8** form a recurring shared-subspace cluster, with heads **4** and **9** also appearing in the high-overlap neighborhood depending on rank. The combined attention-head output is compressible to half width, but less cleanly than the FFN input: **~0.162 relative MSE / ~0.924 cosine** vs **~0.095 / ~0.951** for FFN input after the same 10-epoch linear autoencoder run. Independent per-head half-width autoencoders are much worse (**~0.350 / ~0.856**) despite using the same total bottleneck width (`16 * 64 = 1024`), so the useful compression appears to come from allowing the bottleneck to mix information across heads. Naive deeper GELU MLP autoencoders underperform the linear versions at the same bottleneck and 10-epoch budget. Linear-initialized residual decoders are better: they start exactly at the linear baseline and improve combined-head reconstruction to **~0.149 / ~0.931** and per-head reconstruction to **~0.325 / ~0.869**. Training the combined-head residual decoder **through frozen `o_proj`** gives the best residual-space result so far (**~0.0765 post-`o_proj` rel MSE / ~0.964 cosine**) while slightly sacrificing raw pre-`o_proj` reconstruction, which is the expected tradeoff. Distilling `o_proj(AE(x))` into a replacement `o_proj` works nearly exactly for dense, but low-rank `o_proj` is a quality/parameter tradeoff: rank 768 is much better than rank 512, but still adds enough error that it is worse against the original teacher than the AE target itself. More data helps only modestly for low-rank `o_proj`; rank/capacity is the bigger limiter. Supervised-PCA initialization helps, especially at rank 768, and mostly helps immediately at init rather than through later training.

Remote artifacts:

```text
/mnt/sdb1/dolma-v1_6-sample/layer00_internals_64pb
/mnt/sdb1/dolma-v1_6-sample/layer00_internals_160pb
/mnt/sdb1/dolma-v1_6-sample/layer00_head_pca_ffn_ae_half
/mnt/sdb1/dolma-v1_6-sample/layer00_head_concat_ae_half
/mnt/sdb1/dolma-v1_6-sample/layer00_head_concat_ae_half_mlp
/mnt/sdb1/dolma-v1_6-sample/layer00_head_concat_ae_half_residual_mlp
/mnt/sdb1/dolma-v1_6-sample/layer00_head_concat_ae_half_oproj_loss
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_dense
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_lowrank512
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_lowrank768
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_lowrank512_160pb
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_lowrank768_160pb
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_pcalowrank512_160pb
/mnt/sdb1/dolma-v1_6-sample/layer00_oproj_mimic_ae_pcalowrank768_160pb
/mnt/sdb1/dolma-v1_6-sample/layer00_per_head_output_ae_half
/mnt/sdb1/dolma-v1_6-sample/layer00_per_head_output_ae_half_mlp
/mnt/sdb1/dolma-v1_6-sample/layer00_per_head_output_ae_half_residual_mlp
/mnt/sdb1/dolma-v1_6-sample/layer00_head_subspace_overlap
```

## End-to-end smoke (`scripts/17_end_to_end_smoke.py`)

Minimal pipeline on a **synthetic** six-bin CSV + token shard: **`07`** activation hook smoke → **`09`** one-head capture → **`16`** one-epoch Q/K+dense-V train. Uses **`torch` + `transformers`**; writes under `--output-dir`.

## Script index (compression / deployment)

| Script | Role |
| --- | --- |
| `scripts/17_end_to_end_smoke.py` | Tiny fixture → 07 → 09 → 16 |
| `scripts/18_qk_dense_v_model_surgery_smoke.py` | Full-layer patched loss vs baseline (optional `--allow-missing-layers`) |
| `scripts/_qk_surgery_lib.py` | Shared patching, fused low-rank Q/K forward, optional **dense Q/K materialization** |
| `scripts/19_qk_dense_v_compare_outputs.py` | Prompt generations + optional eval JSON |
| `scripts/20_qk_dense_v_finetune.py` | LM fine-tune **Q/K only** after patch; writes new checkpoint tree |
| `scripts/21_inference_speed_benchmark.py` | Prefill + decode timing; `--materialize-dense-qk`, `--no-fused-qk`, `--compile`, CUDAGraph toggles; optional **`--fallback-joint-qkv-root`** |
| `scripts/22_train_sealed_compressed_block.py` | Whole-layer distillation into a **D -> compressed internals -> D** attention+FFN replacement |
| `scripts/23_train_reduced_qk_dim.py` | PCA/gated prune test for true smaller dense Q/K score dimension |
| `scripts/24_train_shared_qk_metric.py` | Shared lower-dimensional Q/K metric-space experiment |
| `scripts/25_train_dense_small_qk_logits.py` | Direct logit training for one-GEMM smaller dense Q/K |
| `scripts/26_triton_lowrank_qk_benchmark.py` | Synthetic Triton fused low-rank projection benchmark |
| `scripts/27_capture_layer_internals.py` | Capture all head contexts plus FFN input for one layer |
| `scripts/28_head_pca_ffn_autoencoder.py` | Head-context PCA spectra and FFN-input half-width autoencoder |
| `scripts/29_head_subspace_overlap.py` | Compare head PCA subspace overlap after per-head `o_proj` lift |
| `scripts/30_per_head_output_autoencoders.py` | Independent half-width autoencoders for each head output |
| `scripts/31_train_compressed_oproj_from_bottleneck.py` | Train dense/low-rank `o_proj` replacements to mimic `o_proj(AE(heads))` |
| `scripts/32_train_bottleneck_ffn_after_mimic_oproj.py` | Co-train bottleneck FFN + mimic `o_proj` on `R_mimic = ffn_input - mimic(heads)` vs teacher MLP; `--loss-kind` `relative` \| `cosine` \| `both` + `--loss-relative-weight` |
| `scripts/33_run_bottleneck_pipeline_all_layers.py` | Per-layer pipeline: capture → mimic → FFN sweep |
| `scripts/34_sweep_bottleneck_ffn_loss.py` | Multi-layer FFN loss sweep; `--both-only`, `--both-weights`, `--run-tag` for separate dirs / master JSON |
| `scripts/35_wait_then_midlayer_ffn_grid.sh` | After main job: `both(w)` grid on layers **1–4** (default weights coarsened in-repo to **`midlayer_wgrid_v2`**; dense historical list kept in comment) |
| `scripts/36_select_ffn_loss_tradeoff.py` | Rank sweep runs by `relative_mse + gamma*(1-cosine)`; optional cosine/MSE gates; `--write-json` |
| `scripts/37_emit_refine_ffn_weights.py` | From **36**’s JSON, emit a tighter `--both-weights` list around per-layer best `w` |

## Bottleneck FFN after mimic `o_proj` (script 32 loss sweep)

Setup (remote, Dolma `160` windows/bin captures, low-rank mimic **`o_proj` rank 768**, same half-width head AE state as earlier work):

- Per layer: `layerNN_ffn_loss_sweep/sweep_summary.json` under `/mnt/sdb1/dolma-v1_6-sample/`.
- Master: `all_layers_ffn_loss_sweep_summary.json` (16 layers after the post-outage resume completed **L10–L15** and merge).

**Eval metrics** (last epoch in each run): **`relative_mse`**, mean **directional `cosine`** between student and teacher FFN output on the held-out windows.

### `loss_relative` vs `loss_cosine` vs `both(w)`

- **`loss_relative`** can achieve **small `relative_mse`** on some mid layers while **eval cosine stays poor (~0.5–0.56)** on layers **1–4**, and is also weak on **12** and **15** in the main sweep. Optimizing relative MSE alone does **not** imply alignment in residual direction.
- **`loss_cosine`** keeps **cosine near 0.9996+** on all layers; **`relative_mse`** is often **≫ 1** on eval because that loss does not target MSE.
- **`loss_both`** mixes the two: small **`w`** (relative weight) favors cosine; larger **`w`** pulls **`relative_mse` down** at some cosine cost.

### Single default `w` across layers (main sweep, `{0.1, 0.25, 0.5}` only)

Using **`score = relative_mse + 80 * (1 - cosine)`** (same spirit as **`36 --gamma 80`**), **`w = 0.1`** wins **15 / 16** layers; **only layer 7** prefers **`w = 0.25`**. So under a **cosine-heavy** scalar, **`0.1`** is slightly better on average.

**Practical read:** **`w = 0.25`** still looks **strong by eye** on many layers: **cosine stays in the high 0.9994+** band while **`relative_mse`** is often **lower than at `0.1`**. It is a reasonable **one-knob compromise** if you care a bit more about MSE than the `gamma=80` scorer does; if the hard requirement is **cosine as close to 1 as possible**, favor **`0.1`** or run **`36`** with a **larger `gamma`** (or a **cosine floor**) to pick per layer.

### Dense mid-layer `both(w)` grid (`midlayer_wgrid_v1`, layers **1–4** only)

Completed on remote: **`layerNN_ffn_loss_sweep_midlayer_wgrid_v1/sweep_summary.json`**. Thirteen weights from **`0.03`** to **`0.5`**.

Among **`loss_both_w*`** runs, picking **highest cosine** then **lower MSE** gives:

| Layer | Best run | `relative_mse` | `cosine` |
| ---: | --- | ---: | ---: |
| 1 | `loss_both_w0p03` | ~0.00256 | ~0.999724 |
| 2 | `loss_both_w0p03` | ~0.00153 | ~0.999805 |
| 3 | `loss_both_w0p15` | ~0.00243 | ~0.999772 |
| 4 | `loss_both_w0p05` | ~0.00213 | ~0.999847 |

For a **single scalar** pick from that whole sweep, use **`scripts/36_select_ffn_loss_tradeoff.py --run-tag midlayer_wgrid_v1`** with your chosen **`--gamma`** and filters.

### Ops notes

- **Power outage (May 2026):** layers **10–15** of the main sweep were re-run; master summary was **merged** with the pre-outage **0–9** backup; chained **`35`** then ran the mid-layer dense grid until **`RESUME PIPELINE ALL DONE`** on the host.
- **`scripts/35_wait_then_midlayer_ffn_grid.sh`** default **`--both-weights`** was **coarsened** (7 points) and **`--run-tag midlayer_wgrid_v2`** so a shorter rerun does not collide with **`v1`** artifacts; the script comment retains the original dense list.

Remote artifacts (FFN sweep):

```text
/mnt/sdb1/dolma-v1_6-sample/all_layers_ffn_loss_sweep_summary.json
/mnt/sdb1/dolma-v1_6-sample/layerNN_ffn_loss_sweep/
/mnt/sdb1/dolma-v1_6-sample/layerNN_ffn_loss_sweep_midlayer_wgrid_v1/
```

## Tests

`tests/test_qk_surgery_fusion.py` checks **fused vs naive** `MultiHeadQKLowRankProjection` outputs, mixed-rank fallback, and **materialized `Linear` vs fused** equivalence. Requires `pytest` where you run tests.
