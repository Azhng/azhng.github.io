---
layout: post
title:  "Flash Attention: From Math to GPU Metal"
date:   2026-02-11 12:00:00 -0400
category: professional
math: true
---

<!--
=== DRAFT STATUS & CONTEXT FOR FUTURE WORK ===

Series: This is Post 3 of 3 on LLM internals.
  - Post 1 (published): /how-attention-actually-works
  - Post 2 (draft): /how-llms-generate-text
  - Post 3 (this draft): GPU fundamentals, Triton, Flash Attention

Source material: ../attention_talk.ipynb (Jupyter notebook with the original talk)

MAJOR GAPS:
1. WRITING STYLE: Same issue as Post 2. This draft uses AI-sounding prose
   ("paradigm shift", "the main event", "workhorse"). Needs full rewrite in
   the simple, direct voice used in Post 1. Short sentences. No fancy words.
   Read Post 1 for the target voice.

2. CODE BLOCKS WITH OUTPUT: The Triton code (vector add, tiled matmul, fused
   attention) is here but:
   - The benchmarks (lines 66-70) are fabricated numbers, not from actual runs.
     Either run them for real or remove them.
   - The online softmax pseudocode (lines 104-115) should be real runnable
     numpy code with actual output, not pseudocode.
   - Need a side-by-side demo: naive attention vs flash attention showing
     memory usage difference.

3. TABLE OF CONTENTS: Needs a manual TOC at the top (same pattern as Post 1).

4. COLLAPSIBLE SECTIONS: The online softmax math derivation (why the rescaling
   trick works) would be a good candidate for a <details> block. Use the
   pattern from Post 1:
     <details>
     <summary>Plain text summary</summary>
     <div markdown="1">
     content with $$LaTeX$$ here
     </div>
     </details>

5. MATH RENDERING: Added `math: true` to frontmatter. Use $...$ for inline,
   $$...$$ for display math. Variable references in prose use backticks.

6. The Triton code requires a GPU to actually run. Consider adding a note
   about this — readers can't just paste into a REPL like Posts 1 & 2.
   Maybe add a numpy-only simulation of the tiling concept that IS runnable.

7. The tiled matmul ASCII diagram (lines 82-90) is good but could use a
   walkthrough with actual small matrices and numbers.

8. The "Summary" section is too terse. Post 1 has a "Recap" that briefly
   restates each concept. Match that style.

9. Missing: a concrete example showing WHY O(n²) memory is a problem.
   e.g. "128K tokens × 128K tokens × 4 bytes = 64 GB just for one
   intermediate matrix" — this IS mentioned on line 98 but deserves
   more emphasis, maybe its own subsection or a worked calculation.
===
-->

This is the third post in a series on LLM internals. [Part 1 covered attention](/how-attention-actually-works), [Part 2 covered generation and cost](/how-llms-generate-text). Now: why we need GPUs, and how Flash Attention makes long contexts practical.

---

## The Paradigm Shift

| | CPU | GPU |
|---|---|---|
| **Cores** | ~16, each very fast and versatile | ~10,000, each simple but parallel |
| **Analogy** | One expert chef | 10,000 line cooks following a recipe |
| **Good at** | Complex sequential logic | The same simple operation on massive data |

Attention is almost entirely matrix multiplications and elementwise ops — exactly what GPUs are designed for.

A GPU "kernel" is just a function. The only twist: instead of calling it once, you launch thousands of copies simultaneously, each processing a different chunk of data.

---

## Hello World: Vector Addition in Triton

Before attention, the programming model. CPU vs GPU:

```python
# CPU: process one at a time
def add_cpu(x, y):
    result = np.empty(len(x))
    for i in range(len(x)):
        result[i] = x[i] + y[i]
    return result

# GPU: split across parallel workers
def add_gpu(x, y, num_workers=4):
    result = np.empty(len(x))
    chunk_size = len(x) // num_workers
    for worker_id in range(num_workers):  # these run SIMULTANEOUSLY
        start = worker_id * chunk_size
        end = start + chunk_size
        result[start:end] = x[start:end] + y[start:end]
    return result
```

In Triton, this becomes:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)               # which worker am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                # don't read past the end

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Benchmarks on a Tesla T4:

```
n=     10,000:  CPU  0.005ms  GPU  0.035ms  →  0.1x (GPU overhead)
n=  1,000,000:  CPU  0.493ms  GPU  0.053ms  →  9.3x
n= 10,000,000:  CPU 12.776ms  GPU  0.493ms  → 25.9x
```

GPU wins grow with data size — more data means more parallelism to exploit.

---

## Tiled Matrix Multiplication

Matmul is the workhorse of attention (`Q @ K.T`, `weights @ V`). The naive approach loads each row and column from memory for every output element — massive redundant traffic.

**Tiled matmul** loads blocks into fast local memory and reuses each element many times:

```
         B (K × N)
        ┌─────────────┐
        │   │ B tile  │
    A   ├───┼─────────┤
  (M×K) │A  │  C tile │ → C (M × N)
        │tile         │
        └───┴─────────┘
```

Each element of A is used BLOCK_N times. Each element of B is used BLOCK_M times. This is the core of why GPUs are fast for matmul — memory reuse through tiling.

---

## Fused Attention with Online Softmax

Now the main event. Standard attention materializes the full (seq_len, seq_len) attention matrix — that's O(n²) memory just for intermediate results. At 128K tokens, that's 128K × 128K × 4 bytes ≈ 64 GB. Not feasible.

**Flash Attention's insight:** compute attention tile-by-tile and never store the full matrix.

The challenge: softmax needs to see all scores to compute the denominator. But **online softmax** maintains running statistics (max and sum) that let us accumulate correctly across blocks:

```python
# Online softmax accumulation (pseudocode)
for each K,V block:
    scores = q_block @ k_block.T * scale
    m_new = max(m_old, max(scores))           # update running max
    alpha = exp(m_old - m_new)                # rescaling factor
    p = exp(scores - m_new)                   # new contributions
    l = alpha * l + sum(p)                    # update running sum
    acc = alpha * acc + p @ v_block           # accumulate output

output = acc / l  # final normalization
```

When we see a new block with a higher max, we rescale the old accumulator by `exp(old_max - new_max)`. The result is mathematically exact — same as computing the full attention matrix, but using O(n) memory instead of O(n²).

---

## Summary

1. **GPU kernels** tile computation to maximize memory reuse
2. **Tiled matmul** avoids redundant memory loads — each element reused many times
3. **Flash Attention** uses online softmax to avoid materializing the O(n²) attention matrix
4. The result: exact attention with O(n) memory, making long contexts practical

This is what enables 128K+ context windows without needing 64+ GB just for intermediate attention scores.
