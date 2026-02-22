---
layout: post
title:  "How Flash Attention Works"
date:   2026-02-21 12:00:00 -0500
category: professional
math: true
---

This is the third post in a series on LLM internals. [Part 1 covered attention](/how-attention-actually-works). [Part 2 covered generation](/how-llms-generate-text). This post: the Flash Attention algorithm and why it matters for long contexts.

Standard attention creates a score matrix that's `(seq_len, seq_len)`. That's O(n²) memory. At 128K tokens, it's ~64 GB — for one attention head. [Flash Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) avoids this by tiling the computation into small blocks and using an online softmax trick to never build the full matrix. The result is O(n) memory and the same exact output.

There are now multiple versions — [v2](https://arxiv.org/abs/2307.08691) improved GPU utilization with better parallelism, [v3](https://arxiv.org/abs/2407.08691) added async memory ops and FP8 for H100s, and [v4](https://tridao.me/blog/2025/flash4/) targets Blackwell GPUs with deeper pipelining. The core algorithm — tiling + online softmax — is the same across all of them. The versions differ in how they map that algorithm to GPU hardware. This post covers the algorithm.

All code here is numpy — runnable in a Python REPL, no GPU required. Fair warning: the online softmax section gets into some math. Not hard math, but there's a derivation I had to work through step by step before it clicked.

**Contents**
- [The O(n²) memory problem](#the-on2-memory-problem)
- [Standard attention](#standard-attention)
- [Online softmax](#online-softmax)
- [Flash Attention](#flash-attention)
- [Memory scaling](#memory-scaling)
- [Correctness at scale](#correctness-at-scale)
- [Causal masking](#causal-masking)
- [Recap](#recap)

---

## The O(n²) memory problem {#the-on2-memory-problem}

Standard attention computes a `(seq_len, seq_len)` score matrix. Each element is FP32 (4 bytes). How big does that get?

```
   Context    Score matrix              Memory
     1,024    1,024 × 1,024               4 MB
     4,096    4,096 × 4,096              67 MB
    16,384    16,384 × 16,384           1.1 GB
    65,536    65,536 × 65,536          17.2 GB
   131,072    131,072 × 131,072        68.7 GB
```

At 128K tokens, the score matrix alone is ~64 GB. An A100 has 80 GB total. And this is one intermediate result, for one attention head.

---

## Standard attention {#standard-attention}

From [Part 1](/how-attention-actually-works) — the version that creates the full score matrix:

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def standard_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)     # (seq_len, seq_len) — the O(n²) matrix
    weights = softmax(scores, axis=-1)   # (seq_len, seq_len)
    output = weights @ V                 # (seq_len, d_v)
    return output

np.random.seed(42)
seq_len, d_k = 8, 4
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output_std = standard_attention(Q, K, V)
scores = Q @ K.T / np.sqrt(d_k)
print(f"Q, K, V:  ({seq_len}, {d_k}) each")
print(f"Scores:   {scores.shape} — this is the O(n²) matrix")
print(f"Output:   {output_std.shape}")
```

```
Q, K, V:  (8, 4) each
Scores:   (8, 8) — this is the O(n²) matrix
Output:   (8, 4)
```

Three steps. Two intermediate matrices of size `(seq_len, seq_len)`. In this implementation, both live in memory at the same time.

---

## Online softmax {#online-softmax}

Flash Attention depends on being able to compute softmax without seeing all the scores at once. Here's how that works.

Softmax of a vector $x$:

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

where $m = \max(x)$. Computing even one output value needs the max and sum over ALL of $x$.

```python
def naive_softmax(x):                    # x: (n,)
    """Two-pass softmax. Needs the full vector upfront."""
    m = np.max(x)                        # scalar — pass 1: find max
    exp_x = np.exp(x - m)               # (n,)
    total = np.sum(exp_x)               # scalar — pass 2: find sum
    return exp_x / total                # (n,)

x = np.array([1.0, 3.0, 0.5, 2.0, 5.0, 1.5])
print(f"x:       {x}")
print(f"softmax: {naive_softmax(x)}")
```

```
x:       [1.  3.  0.5 2.  5.  1.5]
softmax: [0.0147 0.1087 0.0089 0.04   0.8034 0.0243]
```

Computing `softmax(x[0])` needed `max(x)` and `sum(exp(x - m))`. Both require the entire vector. No way to start outputting early.

**Online softmax** processes $x$ in chunks, maintaining a running max and running sum:

$$m_{\text{new}} = \max(m_{\text{old}},\; \max(\text{chunk}))$$

$$\text{running_sum} \leftarrow \text{running_sum} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum_{i \in \text{chunk}} e^{x_i - m_{\text{new}}}$$

When a new chunk has a higher max, the old sum gets multiplied by $e^{m_{\text{old}} - m_{\text{new}}}$. Since $m_{\text{new}} \geq m_{\text{old}}$, this factor is $\leq 1$ — old terms shrink, which makes sense when a bigger value shows up.

<details>
<summary>Why the rescaling works (derivation)</summary>
<div markdown="1">

The running sum before this update is:

$$\text{running_sum} = \sum_{i \in \text{seen}} e^{x_i - m_{\text{old}}}$$

Each term was computed relative to $m_{\text{old}}$. Multiplying by the rescaling factor rebases each term to $m_{\text{new}}$:

$$e^{x_i - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} = e^{(x_i - m_{\text{old}}) + (m_{\text{old}} - m_{\text{new}})} = e^{x_i - m_{\text{new}}}$$

This uses $e^{a+b} = e^a \cdot e^b$. The $m_{\text{old}}$ terms cancel.

Every term in the sum shares the same $m_{\text{old}}$, so the factor pulls out:

$$\left(\sum_{i \in \text{seen}} e^{x_i - m_{\text{old}}}\right) \cdot e^{m_{\text{old}} - m_{\text{new}}} = \sum_{i \in \text{seen}} e^{x_i - m_{\text{new}}}$$

Now all old terms are relative to $m_{\text{new}}$. Add the new chunk's terms:

$$\underbrace{\text{running_sum} \cdot e^{m_{\text{old}} - m_{\text{new}}}}_{\text{old terms, rebased}} + \underbrace{\sum_{i \in \text{chunk}} e^{x_i - m_{\text{new}}}}_{\text{new terms}} = \sum_{i \in \text{all seen}} e^{x_i - m_{\text{new}}}$$

The result is exact.

</div>
</details>

```python
def online_softmax(x, block_size=3):          # x: (n,)
    """Compute softmax in chunks. Never sees the full vector at once."""
    running_max = -np.inf                     # scalar
    running_sum = 0.0                         # scalar — sum of exp(x_i - running_max)

    for start in range(0, len(x), block_size):
        block = x[start:start+block_size]     # (≤block_size,)
        new_max = max(running_max, np.max(block))

        # Rescale old sum to new max, add new contributions
        running_sum = (running_sum * np.exp(running_max - new_max)
                       + np.sum(np.exp(block - new_max)))
        running_max = new_max

    # NOTE: this line still touches all of x — just for demo.
    # Flash Attention avoids this by accumulating weights @ V inside the loop.
    result = np.exp(x - running_max) / running_sum  # (n,)
    return result

x = np.array([1.0, 3.0, 0.5, 2.0, 5.0, 1.5])
naive_result = naive_softmax(x)
online_result = online_softmax(x, block_size=3)
print(f"Naive softmax:  {naive_result}")
print(f"Online softmax: {online_result}")
print(f"Max difference: {np.max(np.abs(naive_result - online_result)):.2e}")
```

```
Naive softmax:  [0.0147 0.1087 0.0089 0.04   0.8034 0.0243]
Online softmax: [0.0147 0.1087 0.0089 0.04   0.8034 0.0243]
Max difference: 0.00e+00
```

Same result. But online softmax computed the max and sum in two chunks of 3, without needing the full vector for those.

One thing to notice: the last line of `online_softmax` — `result = np.exp(x - running_max) / running_sum` — still touches the full vector. This is just a demo of the algorithm. Flash Attention avoids this by building the output incrementally inside the loop, as shown next.

---

## Flash Attention {#flash-attention}

Standard attention treats softmax as a black box:

$$\text{output} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$

Written this way, softmax has to finish before `@ V` starts. Expanding the softmax shows why that's not actually necessary.

**What does `@ V` do?** For query row $i$, the output is a weighted sum of all $V$ rows:

$$\text{output}_i = \sum_j w_{ij} \cdot V_j$$

Each $V_j$ is a `(d_v,)` vector. $w_{ij}$ is a scalar — "how much should query $i$ attend to key $j$." The result $\text{output}_i$ is a `(d_v,)` vector, one row of the `(seq_len, d_v)` output matrix. (In the demo code, $d_k = d_v$ for simplicity.)

**What's inside the weight?** It's the softmax of the raw score between query $i$ and key $j$:

$$\text{scores}[i,j] = \frac{q_i^T k_j}{\sqrt{d_k}} = \frac{\sum_{l=1}^{d_k} q_{i,l} \cdot k_{j,l}}{\sqrt{d_k}}$$

$$w_{ij} = \frac{e^{\text{scores}[i,j] - m_i}}{\sum_k e^{\text{scores}[i,k] - m_i}}$$

**Substitute back** and notice the denominator is the same for every $j$ in the sum. It pulls out:

$$\text{output}_i = \frac{\overbrace{\sum_j e^{\text{scores}[i,j] - m_i} \cdot V_j}^{\text{numerator: weighted sum of V rows}}}{\underbrace{\sum_j e^{\text{scores}[i,j] - m_i}}_{\text{denominator: just a scalar}}}$$

The numerator is "sum up all $V$ rows, each scaled by its unnormalized weight." The denominator is the normalizing constant from softmax — just a number.

Since the denominator is a scalar, numerator and denominator can be accumulated separately, tile by tile. Compute a chunk of scores, multiply by the corresponding $V$ rows, add to a running total. Keep a running denominator. Divide once at the end. The full score matrix never needs to exist.

### Tiling

Instead of computing the full `(seq_len, seq_len)` score matrix, Flash Attention processes small `(BLOCK, BLOCK)` tiles:

![Standard attention creates the entire score matrix. Flash attention computes one small tile at a time, updates the running accumulators, then discards the tile.](/assets/imgs/flash-attention-tiling.png)

Left: standard attention creates the entire 8×8 score matrix. Right: flash attention computes one 2×2 tile at a time, updates the running accumulators, then discards it.

### The code

```python
def flash_attention(Q, K, V, BLOCK_SIZE=2):
    """Flash Attention in numpy. Same result as standard attention, O(n) memory.

    Uses Q-outer loop here because it reads naturally: "for each query,
    iterate over all keys." The real Flash Attention on GPU uses K/V-outer
    because it controls which data stays in fast SRAM.
    """
    seq_len, d_k = Q.shape
    scale = 1.0 / np.sqrt(d_k)

    # O accumulates the NUMERATOR: sum of exp(score - max) * V, tile by tile.
    # running_sum accumulates the DENOMINATOR: sum of exp(score - max).
    # At the end, O / running_sum = the full attention output.
    O = np.zeros_like(V)                              # (seq_len, d_k) — numerator
    running_max = np.full(seq_len, -np.inf)           # (seq_len,)
    running_sum = np.zeros(seq_len)                   # (seq_len,) — denominator

    for q_start in range(0, seq_len, BLOCK_SIZE):
        q_end = min(q_start + BLOCK_SIZE, seq_len)
        Q_block = Q[q_start:q_end]                    # (block, d_k)

        for kv_start in range(0, seq_len, BLOCK_SIZE):
            kv_end = min(kv_start + BLOCK_SIZE, seq_len)
            K_block = K[kv_start:kv_end]              # (block, d_k)
            V_block = V[kv_start:kv_end]              # (block, d_k)

            # Step 1: scores for this tile
            S_tile = Q_block @ K_block.T * scale      # (block, block) — NOT (seq_len, seq_len)

            # Step 2: update running max
            old_max = running_max[q_start:q_end].copy()                  # (block,)
            new_max = np.maximum(old_max, np.max(S_tile, axis=1))        # (block,)

            # Step 3: rescale old accumulators to new max
            rescale = np.exp(old_max - new_max)                          # (block,)

            # Step 4: unnormalized softmax weights for this tile
            weights_tile = np.exp(S_tile - new_max[:, None])             # (block, block)

            # Step 5: update denominator (same rescaling as online softmax)
            running_sum[q_start:q_end] = (rescale * running_sum[q_start:q_end]
                                          + np.sum(weights_tile, axis=1))

            # Step 6: update numerator — rescale rebases old weights to new_max,
            # then add this tile's contribution (weights_tile @ V_block)
            O[q_start:q_end] = (rescale[:, None] * O[q_start:q_end]
                                + weights_tile @ V_block)

            running_max[q_start:q_end] = new_max

    # Step 7: numerator / denominator = final attention output
    O = O / running_sum[:, None]
    return O

output_flash = flash_attention(Q, K, V, BLOCK_SIZE=2)
diff = np.max(np.abs(output_std - output_flash))
print(f"Standard attention score matrix: {seq_len}×{seq_len} = {seq_len**2} elements")
print(f"Flash attention largest tile:    2×2 = 4 elements")
print(f"Memory ratio:                   {seq_len**2 // 4}× less")
print(f"\nMax difference from standard:   {diff:.2e}")
```

```
Standard attention score matrix: 8×8 = 64 elements
Flash attention largest tile:    2×2 = 4 elements
Memory ratio:                   16× less

Max difference from standard:   2.22e-16
```

Matches standard attention to machine epsilon, and the full `(seq_len, seq_len)` matrix was never allocated.

---

## Memory scaling {#memory-scaling}

The biggest memory difference is the score buffer. Standard attention allocates the full `(n, n)` score matrix. Flash attention only ever holds one `(BLOCK, BLOCK)` tile:

```python
BLOCK = 64  # typical block size
print(f"{'seq_len':>10}  {'Standard (n×n scores)':>25}  {'Flash (BLOCK×BLOCK tile)':>26}  {'Ratio':>8}")
print("-" * 74)
for n in [512, 2048, 8192, 32768, 131072]:
    std_bytes = n * n * 4
    flash_bytes = BLOCK * BLOCK * 4
    ratio = std_bytes / flash_bytes

    def fmt(b):
        if b >= 1e9: return f"{b/1e9:.1f} GB"
        if b >= 1e6: return f"{b/1e6:.1f} MB"
        return f"{b/1e3:.1f} KB"

    print(f"{n:>10,}  {fmt(std_bytes):>25}  {fmt(flash_bytes):>26}  {ratio:>7.0f}×")
```

```
   seq_len      Standard (n×n scores)    Flash (BLOCK×BLOCK tile)     Ratio
--------------------------------------------------------------------------
       512                     1.0 MB                    16.4 KB       64×
     2,048                    16.8 MB                    16.4 KB     1024×
     8,192                   268.4 MB                    16.4 KB    16384×
    32,768                     4.3 GB                    16.4 KB   262144×
   131,072                    68.7 GB                    16.4 KB  4194304×
```

This is just the score buffer — both approaches still need Q, K, V, and the output matrix. But those are all O(n), so they're the same. The score matrix is the O(n²) term that dominates, and that's what Flash Attention eliminates.

---

## Correctness at scale {#correctness-at-scale}

Flash attention matches standard attention across block sizes, on a larger example (64 tokens, 32 dimensions):

```python
np.random.seed(123)
n, d = 64, 32
Q2 = np.random.randn(n, d)
K2 = np.random.randn(n, d)
V2 = np.random.randn(n, d)

out_std = standard_attention(Q2, K2, V2)

for block_size in [2, 4, 8, 16, 32]:
    out_flash = flash_attention(Q2, K2, V2, BLOCK_SIZE=block_size)
    diff = np.max(np.abs(out_std - out_flash))
    print(f"  BLOCK_SIZE={block_size:>2}: max diff = {diff:.2e}  {'PASS' if diff < 1e-6 else 'FAIL'}")
```

```
  BLOCK_SIZE= 2: max diff = 5.55e-16  PASS
  BLOCK_SIZE= 4: max diff = 6.11e-16  PASS
  BLOCK_SIZE= 8: max diff = 5.55e-16  PASS
  BLOCK_SIZE=16: max diff = 5.27e-16  PASS
  BLOCK_SIZE=32: max diff = 4.44e-16  PASS
```

All differences are at floating-point epsilon. The tiling is invisible to the output.

---

## Causal masking {#causal-masking}

In autoregressive models (GPT-style), token $i$ can only attend to tokens $\leq i$. The score matrix has a lower-triangular structure — everything above the diagonal is $-\infty$ before softmax, so future tokens get zero weight. ([Part 1](/how-attention-actually-works) covered this.)

With tiling, causal masking adds two things:

1. **Skip entire tiles** where the K/V block is entirely in the future. The Q-outer loop handles this naturally: the K/V inner loop runs to `q_end` instead of `seq_len`.
2. **Mask within tiles** that straddle the diagonal. Set future positions to $-\infty$ before computing softmax weights.

```python
def flash_attention_causal(Q, K, V, BLOCK_SIZE=2):
    """Flash Attention with causal masking.
    Two additions vs the unmasked version — marked with ← CAUSAL.
    """
    seq_len, d_k = Q.shape
    scale = 1.0 / np.sqrt(d_k)

    O = np.zeros_like(V)                               # (seq_len, d_k) — numerator
    running_max = np.full(seq_len, -np.inf)            # (seq_len,)
    running_sum = np.zeros(seq_len)                    # (seq_len,) — denominator

    for q_start in range(0, seq_len, BLOCK_SIZE):
        q_end = min(q_start + BLOCK_SIZE, seq_len)
        Q_block = Q[q_start:q_end]

        #                          vvvvv ← CAUSAL: stop at q_end, not seq_len
        for kv_start in range(0, q_end, BLOCK_SIZE):
            kv_end = min(kv_start + BLOCK_SIZE, seq_len)
            K_block = K[kv_start:kv_end]
            V_block = V[kv_start:kv_end]

            S_tile = Q_block @ K_block.T * scale

            # ← CAUSAL: mask future positions within tiles on the diagonal
            q_positions = np.arange(q_start, q_end)[:, None]     # (block, 1)
            kv_positions = np.arange(kv_start, kv_end)[None, :]  # (1, block)
            causal_mask = kv_positions > q_positions               # (block, block)
            S_tile = np.where(causal_mask, -np.inf, S_tile)

            # Everything below is identical to the unmasked version
            old_max = running_max[q_start:q_end].copy()
            new_max = np.maximum(old_max, np.max(S_tile, axis=1))

            rescale = np.exp(old_max - new_max)
            weights_tile = np.exp(S_tile - new_max[:, None])

            running_sum[q_start:q_end] = (rescale * running_sum[q_start:q_end]
                                          + np.sum(weights_tile, axis=1))
            O[q_start:q_end] = (rescale[:, None] * O[q_start:q_end]
                                + weights_tile @ V_block)
            running_max[q_start:q_end] = new_max

    O = O / running_sum[:, None]
    return O
```

```python
# Standard causal attention (for comparison)
scores_causal = Q @ K.T / np.sqrt(d_k)
mask = np.triu(np.ones((seq_len, seq_len)), k=1)   # 1s above diagonal
scores_causal = np.where(mask, -np.inf, scores_causal)
weights_causal = softmax(scores_causal, axis=-1)
output_causal_std = weights_causal @ V

output_causal_flash = flash_attention_causal(Q, K, V, BLOCK_SIZE=2)
print(f"Max diff (causal flash vs standard): "
      f"{np.max(np.abs(output_causal_std - output_causal_flash)):.2e}")
```

```
Max diff (causal flash vs standard): 4.44e-16
```

The causal score matrix looks like this — $-\infty$ above the diagonal, real scores below:

```
[[-0.59   -inf   -inf   -inf   -inf   -inf   -inf   -inf]
 [ 0.31 -0.77   -inf   -inf   -inf   -inf   -inf   -inf]
 [-0.19 -0.32 -0.03   -inf   -inf   -inf   -inf   -inf]
 [ 0.64  2.99  0.11  0.61   -inf   -inf   -inf   -inf]
 [ 0.33  0.05 -0.08  0.10 -0.33   -inf   -inf   -inf]
 [ 1.01  0.19  0.73 -1.77  0.74 -1.19   -inf   -inf]
 [-0.76  0.64 -0.18  0.83 -0.45 -0.20  0.20   -inf]
 [-1.22  0.81 -0.49  1.67 -0.30  0.67  1.10 -0.60]]
```

---

## Recap {#recap}

Standard attention creates a `(seq_len, seq_len)` score matrix — ~64 GB at 128K tokens, for one attention head. The bottleneck is softmax: it needs the max and sum over the full row before it can produce any output.

Online softmax gets around this by processing scores in chunks and rescaling the running accumulator by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever a bigger value shows up. The result is exact.

Flash Attention applies this to the full attention computation. The numerator (weighted sum of $V$) and denominator (normalizing constant) are accumulated tile by tile. The largest intermediate is `(BLOCK, BLOCK)` instead of `(seq_len, seq_len)`. For causal masking, two additions: stop the K/V loop early, and mask future positions within diagonal tiles.

The output matches standard attention to floating-point epsilon. All the tiling is invisible from the outside.

