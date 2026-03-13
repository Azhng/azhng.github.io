---
layout: post
title:  "Forcing Flash Attention onto a TPU and Learning the Hard Way"
date:   2026-03-06 12:00:00 -0500
category: professional
math: true
---

This is the fifth post in a series on LLM internals. [Part 1 covered attention](/how-attention-actually-works), [Part 2 covered generation](/how-llms-generate-text), [Part 3 covered the Flash Attention algorithm](/how-flash-attention-works), [Part 4 put it on a GPU with Triton](/how-flash-attention-runs-on-a-gpu). This post takes the Triton kernel from Part 4 and ports it to a TPU.

Part 4 was a lot of work but also a lot of fun. And while working in Colab, I noticed that TPU was offered for free in the free tier. So I figured, why not just take Part 4's flash attention and port it to TPU? I know the algorithm, I've written the kernel, JAX is just "numpy but compiled." Translate, benchmark, call it a day.

It did not go that way.

The code uses [JAX](https://jax.readthedocs.io/) and runs on a TPU. To follow along, a free Colab TPU runtime works (`Runtime -> Change runtime type -> TPU`).

---

**Contents**
- [JAX/XLA: the TPU programming model](#jax-xla)
- [Standard causal attention](#standard-attention)
- [Flash attention in JAX](#flash-attention-jax)
- [Benchmark](#benchmark)
- [What just happened?](#what-just-happened)
- [The vmap insight](#vmap)
- [OK but seriously, what even is a TPU?](#what-is-a-tpu)
- [How data flows through a systolic array](#systolic-array)
- [Building a systolic array emulator](#emulator)
- [What the emulator revealed](#emulator-revealed)
- [Pallas: what it would take to beat the compiler](#pallas)
- [What I actually learned](#what-i-learned)

---

## JAX/XLA: the TPU programming model {#jax-xla}

In Part 4, I wrote Triton kernels: explicit `program_id`, pointer arithmetic, `tl.load`/`tl.store`. The code controls exactly which bytes move where.

JAX is a layer above that. You express operations as `matmul`, `exp`, `where`, and the XLA compiler decides how to map them to hardware. When `jax.jit` is invoked:

1. JAX traces the Python function, running it once with abstract values to record which ops happen
2. The trace becomes HLO (High-Level Operations), a graph of ~100 primitives like `dot`, `reduce`, `broadcast`
3. XLA optimizes. The big one is fusing sequences of elementwise ops into single kernels so intermediates never hit HBM
4. XLA compiles to device code: PTX for GPU, VLIW instructions for TPU

The Python isn't running on the TPU. It's a specification that gets compiled into a static binary.

### Mutability is gone

Triton gives mutable pointers. `tl.store(ptr, val)` writes wherever you want. JAX arrays are immutable. There's no `out[i] = val`.

`jax.jit` traces the function into a pure computation graph, and mutation would create side effects that break tracing. This has concrete consequences for the flash attention loop:

| Triton (Part 4) | JAX (this post) |
|---|---|
| `tl.store(out_ptrs, acc, mask=...)` | `out = lax.dynamic_update_slice(out, tile, (start, 0))` |
| `for kv_start in range(0, q_end, BLOCK_KV):` | `jax.lax.fori_loop(0, num_k_blocks, k_body, state)` |
| Mutable `acc += tl.dot(weights, v)` | Return new state: `return (new_max, new_sum, new_acc)` |
| Pointer arithmetic for tile addresses | Compiler handles data movement |

A note on `jax.lax.fori_loop`: a Python `for` loop gets unrolled at trace time. 100 iterations means 100 copies of the loop body in the computation graph. `fori_loop` tells XLA "this is a loop" so it compiles to an actual hardware loop. The body must be a pure function that takes state in and returns state out.

And `dynamic_update_slice` returns a new array with a slice replaced. "Dynamic" means the start index can be a runtime value (like `q_start`), but the slice size must be known at compile time.

---

## Standard causal attention {#standard-attention}

Same baseline as Parts 3 and 4, materializing the full `(n, n)` score matrix:

```python
def standard_causal_attention(Q: jax.Array, K: jax.Array, V: jax.Array) -> jax.Array:
    """Standard causal attention. Shapes: Q, K, V: (n, d) -> out: (n, d)"""
    assert Q.ndim == K.ndim == V.ndim == 2
    assert Q.shape == K.shape == V.shape

    n, d = Q.shape
    scale = jnp.float32(1.0 / math.sqrt(d))

    q = Q.astype(jnp.float32)
    k = K.astype(jnp.float32)
    v = V.astype(jnp.float32)

    scores = (q @ k.T) * scale                           # (n, n)
    causal_mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    scores = jnp.where(causal_mask, -jnp.inf, scores)   # (n, n)

    weights = jax.nn.softmax(scores, axis=-1)           # (n, n)
    out = weights @ v                                    # (n, d)
    return out.astype(Q.dtype)

standard_causal_attention_jit = jax.jit(standard_causal_attention)
```

Nothing interesting here. XLA sees the entire expression and fuses it into one optimized kernel, so no intermediate matrices spill to HBM. This is the baseline.

---

## Flash attention in JAX {#flash-attention-jax}

Same algorithm as [Part 3](/how-flash-attention-works)'s numpy version and [Part 4](/how-flash-attention-runs-on-a-gpu)'s Triton kernel. Same running state (`running_max`, `running_sum`, `acc`), same per-tile update:

$$new\_max = \max(running\_max,\, \max(\text{scores}))$$

$$rescale = e^{running\_max - new\_max}$$

$$running\_sum \leftarrow rescale \cdot running\_sum + \textstyle\sum e^{\text{scores} - new\_max}$$

$$acc \leftarrow rescale \cdot acc + e^{\text{scores} - new\_max} \cdot V$$

The algorithm is identical. What changes is how JAX's functional model shapes the code.

```python
@partial(jax.jit, static_argnames=("block_m", "block_n"))  # recompiles if block sizes change
def flash_attention_tiled(
    Q: jax.Array, K: jax.Array, V: jax.Array,
    block_m: int = 128, block_n: int = 128,
) -> jax.Array:
    """Causal Flash Attention with tiled online softmax in JAX.
    Same algorithm as Part 3 (numpy) and Part 4 (Triton).
    Lines marked # <-- JAX are where this diverges from the Triton version.
    """
    assert Q.ndim == K.ndim == V.ndim == 2
    assert Q.shape == K.shape == V.shape
    assert block_m > 0 and block_n > 0

    n, d = Q.shape
    q = Q.astype(jnp.float32)
    k_all = K.astype(jnp.float32)
    v_all = V.astype(jnp.float32)
    scale = jnp.float32(1.0 / math.sqrt(d))

    # (row_max, row_sum, accumulator) — the online softmax state
    SoftmaxState = tuple[jax.Array, jax.Array, jax.Array]

    # Pad so every dynamic_update_slice writes a full (block_m, d) chunk.
    # XLA needs static slice sizes — can't write a variable-length chunk.      # <-- JAX
    num_q_blocks = math.ceil(n / block_m)
    num_k_blocks = math.ceil(n / block_n)
    n_pad = num_q_blocks * block_m

    out = jnp.zeros((n_pad, d), dtype=jnp.float32)

    q_offsets = jnp.arange(block_m)
    k_offsets = jnp.arange(block_n)

    # Outer loop over query blocks.
    # fori_loop, not a Python for — otherwise XLA unrolls it at trace time.    # <-- JAX
    def q_body(q_block: int, out_buf: jax.Array) -> jax.Array:
        q_start = q_block * block_m
        q_idx = q_start + q_offsets                                            # (block_m,)
        q_mask = q_idx < n
        q_safe = jnp.minimum(q_idx, n - 1)                   # scalar broadcasts across vector

        q_tile = jnp.where(q_mask[:, None], q[q_safe, :], 0.0)             # (block_m, d)

        # Same running state as Part 3 and Part 4
        running_max = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        running_sum = jnp.zeros((block_m,), dtype=jnp.float32)
        acc = jnp.zeros((block_m, d), dtype=jnp.float32)

        # Inner loop over K/V blocks.
        # State is a tuple — fori_loop body takes it in and returns it out.    # <-- JAX
        def k_body(k_block: int, state: SoftmaxState) -> SoftmaxState:
            running_max, running_sum, acc = state

            k_start = k_block * block_n
            k_idx = k_start + k_offsets                                        # (block_n,)
            k_mask = k_idx < n
            k_safe = jnp.minimum(k_idx, n - 1)               # scalar broadcasts across vector

            k_tile = jnp.where(k_mask[:, None], k_all[k_safe, :], 0.0)     # (block_n, d)
            v_tile = jnp.where(k_mask[:, None], v_all[k_safe, :], 0.0)     # (block_n, d)

            scores = (q_tile @ k_tile.T) * scale                            # (block_m, block_n)

            causal = q_idx[:, None] >= k_idx[None, :]
            valid = q_mask[:, None] & k_mask[None, :] & causal
            scores = jnp.where(valid, scores, -jnp.inf)

            tile_max = jnp.max(scores, axis=1)                              # (block_m,)
            new_max = jnp.maximum(running_max, tile_max)

            rescale = jnp.where(
                jnp.isfinite(running_max),
                jnp.exp(running_max - new_max),
                0.0,
            )
            weights = jnp.where(
                jnp.isfinite(new_max)[:, None],
                jnp.exp(scores - new_max[:, None]),
                0.0,
            )                                                                # (block_m, block_n)

            running_sum = rescale * running_sum + jnp.sum(weights, axis=1)
            acc = rescale[:, None] * acc + weights @ v_tile

            return new_max, running_sum, acc                  # <-- JAX: return new state

        running_max, running_sum, acc = jax.lax.fori_loop(
            0, num_k_blocks, k_body, (running_max, running_sum, acc)
        )

        out_tile = jnp.where(running_sum[:, None] > 0, acc / running_sum[:, None], 0.0)

        # Can't do out_buf[q_start:, :] = out_tile — arrays are immutable.    # <-- JAX
        out_buf = jax.lax.dynamic_update_slice(out_buf, out_tile, (q_start, 0))
        return out_buf

    out = jax.lax.fori_loop(0, num_q_blocks, q_body, out)
    return out[:n, :].astype(Q.dtype)
```

### What tripped me up

The algorithm is the same as Part 4's Triton kernel. Here's what actually changed.

No pointer arithmetic. In Triton, loading a tile meant computing a 2D grid of memory addresses: `A_ptr + offs_row[:, None] * stride + offs_col[None, :]`. In JAX, it's `q[q_safe, :]`, normal array indexing. The compiler figures out the memory access pattern. Easily the biggest readability win.

State goes in, state comes out. In Triton, `acc` is a mutable local variable and `acc += tl.dot(weights, v)` modifies it in place. In JAX, the `fori_loop` body is a pure function: takes `(running_max, running_sum, acc)` as input, returns updated versions. No mutation. I found this awkward at first, but it forces the code to be explicit about what state the loop carries, which is actually nice.

`fori_loop` is not optional. I initially wrote the outer loop as `for q_block in range(num_q_blocks):` and it compiled fine. But XLA unrolled every iteration into the graph and compilation took forever for large sequences. `fori_loop` tells XLA this is a real loop. The body must be a function, and there's no breaking early. Part 4's Triton kernel could stop the KV loop at `q_end` for causal early-stop; here all K blocks get processed and the causal mask zeros out future positions. More wasted compute, but the loop structure stays simple for XLA.

Where do tiles live? In Part 4 I tracked exactly what lived in SRAM vs HBM. In JAX, there's no control over placement. XLA decides what to keep on-chip based on the computation graph. `fori_loop` gives it a hint (`q_tile`, `running_max`, `running_sum`, `acc` are loop-carried state, so XLA will try to keep them on-chip), but that's trusting the compiler rather than specifying it.

`q_offsets` and `k_offsets` are the JAX equivalent of Part 4's `tl.arange`. They create the tile index vectors used for bounds checking and masking. `q_mask = q_idx < n` is the same bounds check that `mask = offsets < n_elements` was in Triton's vector add. `q_safe = jnp.minimum(q_idx, n - 1)` is a clamped gather that prevents out-of-bounds reads, while `q_mask` separately zeros out the garbage values from those clamped positions.

The tradeoff is: Triton gives control, JAX gives portability. The same `flash_attention_tiled` function runs on TPU, GPU, or CPU with zero code changes. You just lose the ability to say "this tile lives in SRAM."

Correctness check (on shapes that aren't multiples of the block size, to test boundary logic):

```
n= 257, d= 64, blocks=(64,64)   match=True  max_abs=0.004399
n= 513, d= 64, blocks=(128,128) match=True  max_abs=0.003483
n= 777, d= 80, blocks=(128,64)  match=True  max_abs=0.005013
```

The max_abs is larger than on GPU. On TPU, XLA may use bf16 internally even when float32 is requested, which gives ~1e-3 precision instead of ~1e-5.

### Memory scaling

Same story as [Part 3](/how-flash-attention-works): the score matrix is O(n²), the output is O(n·d). The flash version never allocates the score matrix:

```
 seq_len    scores (n^2)    output (n*d)       ratio    fits on-chip?
----------------------------------------------------------------------
     512           1.0 MB           0.1 MB         8.0x           yes
    1024           4.0 MB           0.2 MB        16.0x           yes
    2048          16.0 MB           0.5 MB        32.0x           yes
    4096          64.0 MB           1.0 MB        64.0x           yes
    8192         256.0 MB           2.0 MB       128.0x            NO
   16384        1024.0 MB           4.0 MB       256.0x            NO
   32768        4096.0 MB           8.0 MB       512.0x            NO
```

On GPU, the score matrix exceeds SM shared memory (~164 KB) at n=256. On TPU, the on-chip SRAM is ~128 MB, so the score matrix fits until n=8192. That's a 32x higher threshold before tiling becomes strictly necessary for capacity reasons. (More on TPU memory architecture later. These numbers are for a single attention head with d=64. Multi-head attention at d=128 with multiple heads sharing the on-chip memory would shift the crossover point down.)

---

## Benchmark {#benchmark}

On GPU, flash attention was the whole point: avoid materializing the n×n score matrix. On TPU with XLA, standard attention gets auto-fused. Does the tiling actually help?

Setup: all benchmarks run on a Colab TPU v5e (single chip), JAX 0.7.2, float32 inputs, single-head `(n, 64)`. Each timing is the mean of 10 iterations after 1 warmup, measured with `block_until_ready()` to exclude async dispatch. Compilation time excluded.

To simulate "what if XLA didn't fuse" (the GPU-without-Triton experience), I also benchmark an unfused version: three separate jitted functions with `block_until_ready()` between them, forcing each intermediate to materialize in HBM. And a nojit version where every single op is a separate kernel dispatch.

```python
# ── Unfused baseline: simulate GPU-without-Triton on TPU ──────────
# Each step is a separate jitted function. block_until_ready() forces
# each intermediate to materialize in HBM before the next step starts.

@jax.jit
def _step1_scores(q, k, scale, causal_mask):
    scores = (q @ k.T) * scale
    return jnp.where(causal_mask, -jnp.inf, scores)

@jax.jit
def _step2_softmax(scores):
    return jax.nn.softmax(scores, axis=-1)

@jax.jit
def _step3_output(weights, v):
    return weights @ v

def unfused_causal_attention(Q, K, V, causal_mask):
    """Attention with each step as a separate XLA dispatch — no fusion."""
    scale = jnp.float32(1.0 / math.sqrt(Q.shape[-1]))
    scores = _step1_scores(Q, K, scale, causal_mask)
    scores.block_until_ready()          # force HBM round-trip
    weights = _step2_softmax(scores)
    weights.block_until_ready()          # force HBM round-trip
    out = _step3_output(weights, V)
    return out


# ── Maximum suffering: no @jit, every op dispatches separately ────
def nojit_causal_attention(Q, K, V):
    """Every. Single. Op. Is. A. Separate. Kernel. Launch."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = Q @ K.T                                    # dispatch 1
    scores.block_until_ready()
    scores = scores * scale                             # dispatch 2
    scores.block_until_ready()
    mask = jnp.triu(jnp.ones((Q.shape[0], Q.shape[0]), dtype=bool), k=1)
    scores = jnp.where(mask, -jnp.inf, scores)         # dispatch 3
    scores.block_until_ready()
    weights = jax.nn.softmax(scores, axis=-1)           # dispatch 4
    weights.block_until_ready()
    out = weights @ V                                   # dispatch 5
    out.block_until_ready()
    return out
```

```
Backend: tpu
     n   scores(MB)    VMEM?   nojit(ms)   unfused(ms)   fused(ms)   flash(ms)   fuse speedup
-----------------------------------------------------------------------------------------------
   512        1.0      yes       1.390         0.475       0.076       0.082          6.3x
  1024        4.0      yes       1.504         0.497       0.055       0.133          9.0x
  2048       16.0      yes       1.737         0.651       0.067       0.522          9.7x
  4096       64.0      yes       3.016         1.038       0.072       2.509         14.5x
  8192      256.0       NO       7.385         2.834       1.189      14.052          2.4x
 16384     1024.0       NO      25.576        10.110       4.445      89.567          2.3x
 32768     4096.0       NO         OOM           OOM      17.123     103.016             —
```

My flash attention is 35x slower than the fused standard at n=4096. Not a little worse. Catastrophically worse.

Look at the fuse speedup column. The unfused version forces three HBM round-trips (scores, weights, output). The fused version avoids all of them. At n=4096, that's a 14.5x speedup just from fusion. XLA is earning its keep.

The nojit column is there for fun. Every single op (matmul, scale, mask, softmax, final matmul) dispatches as a separate kernel with a full HBM round-trip in between. 3ms at n=4096 vs 0.072ms fused.

---

## What just happened? {#what-just-happened}

Look at those numbers again. My flash attention, the algorithm that was the entire point of Parts 3 and 4, is slower than *unfused* standard attention on TPU at n=4096.

My best theory: the fused standard path wins because XLA sees the entire `softmax(Q @ K.T) @ V` expression at once and compiles it into one optimized kernel with no intermediate matrices spilling to HBM. My flash attention uses `fori_loop`, which XLA likely compiles as a generic sequential loop. It probably can't fuse across iterations, can't pipeline memory loads, can't interleave independent work. (I haven't dumped the HLO to verify this. It's an inference from the benchmark numbers and XLA's documented behavior.)

But the outer loop over Q blocks has zero data dependency between iterations. Each Q block reads the same K/V, maintains its own softmax state, writes to different output rows. The only truly sequential part is the inner K loop, where the running max and sum accumulate tile by tile.

`fori_loop` hides this parallelism from the compiler. XLA does dataflow analysis on the computation graph. If it could *see* that the Q blocks are independent, it could schedule them in parallel, interleave their memory loads, maybe dispatch them to different MXUs.

But `fori_loop` is opaque. It presents as "a loop with carried state." The compiler isn't getting an "these iterations are independent" signal from the code.

What if I just told XLA that the Q tiles have no dependencies on each other?

---

## The vmap insight {#vmap}

`jax.vmap` transforms a function that processes one item into a function that processes a batch. The important part: it tells XLA that every item in the batch is independent. No carried state between them.

Instead of two nested `fori_loop`s, `vmap` replaces the outer Q loop. `fori_loop` stays only for the inner K accumulation, which genuinely is sequential. Same algorithm, same tiles, same math. Just one piece of information the compiler didn't have before.

```python
@partial(jax.jit, static_argnames=("block_m", "block_n"))
def flash_attention_vmap(Q, K, V, block_m=128, block_n=128):
    n, d = Q.shape
    scale = jnp.float32(1.0 / math.sqrt(d))
    num_q_blocks = math.ceil(n / block_m)
    num_k_blocks = math.ceil(n / block_n)
    n_pad = num_q_blocks * block_m

    k_all = K.astype(jnp.float32)
    v_all = V.astype(jnp.float32)
    k_offsets = jnp.arange(block_n)

    # Pad Q and reshape into (num_q_blocks, block_m, d)
    q_padded = jnp.zeros((n_pad, d), dtype=jnp.float32)
    q_padded = q_padded.at[:n, :].set(Q.astype(jnp.float32))
    q_blocks = q_padded.reshape(num_q_blocks, block_m, d)

    q_offsets = jnp.arange(block_m)
    q_starts = jnp.arange(num_q_blocks) * block_m

    # (row_max, row_sum, accumulator) — the online softmax state
    SoftmaxState = tuple[jax.Array, jax.Array, jax.Array]

    def one_q_block(q_tile: jax.Array, q_start: jax.Array) -> jax.Array:
        """Process one Q block against all K/V blocks.
        No data dependency on other Q blocks."""
        q_idx = q_start + q_offsets                                          # (block_m,)
        q_mask = q_idx < n

        running_max = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        running_sum = jnp.zeros((block_m,), dtype=jnp.float32)
        acc = jnp.zeros((block_m, d), dtype=jnp.float32)

        def k_body(k_block: int, state: SoftmaxState) -> SoftmaxState:
            running_max, running_sum, acc = state

            k_start = k_block * block_n
            k_idx = k_start + k_offsets                                      # (block_n,)
            k_mask = k_idx < n
            k_safe = jnp.minimum(k_idx, n - 1)              # scalar broadcasts across vector

            k_tile = jnp.where(k_mask[:, None], k_all[k_safe, :], 0.0)
            v_tile = jnp.where(k_mask[:, None], v_all[k_safe, :], 0.0)

            scores = (q_tile @ k_tile.T) * scale                            # (block_m, block_n)

            causal = q_idx[:, None] >= k_idx[None, :]
            valid = q_mask[:, None] & k_mask[None, :] & causal
            scores = jnp.where(valid, scores, -jnp.inf)

            tile_max = jnp.max(scores, axis=1)
            new_max = jnp.maximum(running_max, tile_max)

            rescale = jnp.where(
                jnp.isfinite(running_max),
                jnp.exp(running_max - new_max),
                0.0,
            )
            weights = jnp.where(
                jnp.isfinite(new_max)[:, None],
                jnp.exp(scores - new_max[:, None]),
                0.0,
            )

            running_sum = rescale * running_sum + jnp.sum(weights, axis=1)
            acc = rescale[:, None] * acc + weights @ v_tile

            return new_max, running_sum, acc

        running_max, running_sum, acc = jax.lax.fori_loop(
            0, num_k_blocks, k_body, (running_max, running_sum, acc)
        )

        out_tile = jnp.where(running_sum[:, None] > 0, acc / running_sum[:, None], 0.0)
        return out_tile

    # vmap over Q blocks — XLA sees all blocks at once, can interleave MXU/VPU/DMA
    all_tiles = jax.vmap(one_q_block)(q_blocks, q_starts)                    # (num_q_blocks, block_m, d)
    out = all_tiles.reshape(n_pad, d)
    return out[:n, :].astype(Q.dtype)
```

Results:

```
fori vs vmap match: True
max diff: 0.000000

     n    fori(ms)    vmap(ms)   fused(ms)   vmap speedup
------------------------------------------------------------
   512       0.074       0.065       0.065          1.1x
  1024       0.133       0.079       0.069          1.7x
  2048       0.525       0.083       0.069          6.3x
  4096       2.510       0.178       0.072         14.1x
  8192      14.061       0.587       1.194         23.9x
 16384      89.538       1.997       4.444         44.8x
```

45x faster at n=16384. Same algorithm. Same tiles. Same math. The only difference: `vmap` instead of `fori_loop` on the outer Q dimension.

The fused column is interesting too. The vmap flash attention doesn't pull ahead until `n=8192`, when the score matrix is 256 MB and no longer fits in ~128 MB of VMEM. At `n=4096`, XLA's fused standard path still wins comfortably. Below that threshold, the fully fused path keeps everything on-chip. Above it, the tiled approach avoids materializing the score matrix entirely. Same win as on GPU, just at a higher threshold because TPU has more on-chip memory.

This was the "aha" moment of the project. The algorithm was never the problem. The compiler just couldn't see the parallelism through `fori_loop`.

---

The practical story is done. The `vmap` fix works, and it beats fused standard attention once the score matrix outgrows VMEM. But I was left wondering *why* the original failed so badly. What is the hardware actually doing with those tiles? The rest of this post is the rabbit hole I went down trying to answer that. Feel free to stop here if the benchmark results are all you wanted.

---

## OK but seriously, what even is a TPU? {#what-is-a-tpu}

The vmap result is wild. 45x faster, and it beats XLA's fused attention at large sizes, just from telling the compiler that Q blocks are independent. But I still don't really understand *why* the original was so slow, or what the hardware is actually doing with those tiles.

### Inside a TPU chip

A TPU v5e chip (what Colab provides in the free tier) has one TensorCore, the unit that does all compute:

```
TPU v5e chip
└── TensorCore
    ├── 4x MXU   (128x128 systolic arrays — the matrix multiply engines)
    ├── 1x VPU   (vector processing unit — elementwise ops, reductions)
    ├── 1x Scalar unit   (control flow, instruction dispatch, DMA orchestration)
    └── ~128 MB VMEM   (shared on-chip SRAM scratchpad)
```

### MXU

On a GPU, the SM is built around CUDA cores, scalar ALUs, 32 of which execute in lockstep as a warp (Part 4 covered this). Tensor cores are a separate thing, specialized matrix multiply units bolted onto each SM. They accelerate matmul, but the SM's general-purpose work still runs on CUDA cores. Tensor cores are an accelerator, not the foundation.

A TPU flips this. The MXU (Matrix Multiply Unit) isn't a bolt-on accelerator, it's the primary compute engine. Each MXU is a 128x128 systolic array: 16,384 multiply-accumulate cells. The v5e has 4 MXUs per chip, all fed from the same VMEM. Everything that can be expressed as a matrix multiply goes through the MXUs.

"Systolic" means data flows through the array rhythmically, like a heartbeat. One matrix is pre-loaded into the cells and stays stationary. The other streams in from the left, flowing through each cell. Every cell multiplies its resident weight by the passing activation, accumulates the partial sum, and hands data to its neighbor. By the time data exits the bottom, you have a full matrix multiply, and no intermediate values touched memory.

### VPU

The VPU (Vector Processing Unit) handles everything that isn't a matmul: elementwise ops (ReLU, exp, add), reductions, type casts. It's a wide SIMD vector unit, think AVX-512 on steroids rather than thousands of CUDA cores.

There's only one VPU shared across the whole chip, and it's roughly 10x slower than the MXUs for the same FLOP count. On TPU, expressing as much computation as matmul as possible matters because everything else is a relative bottleneck.

### No threads

On a GPU, memory latency is hidden by thread parallelism: when one warp stalls on a memory read, the SM switches to another (Part 4 covered this). A TPU has no threads. The scalar unit dispatches instructions to the MXUs and VPU. Latency hiding comes from pipelining: while the MXUs compute one tile, the DMA engine prefetches the next tile from HBM into VMEM. Same goal, completely different mechanism.

For reference, the GPU (A100) has 108 SMs each with 4 tensor cores (432 total), thousands of CUDA cores, and ~164 KB shared memory per SM. Execution is massively threaded with warp switching for latency hiding. The TPU (v5e) has 1 TensorCore with 4 MXUs (128x128 systolic arrays), 1 VPU, and ~128 MB of shared VMEM. Execution is single-threaded and pipelined, with DMA/compute overlap for latency hiding.

### Memory hierarchy

Same structure as GPU (fast on-chip, slow off-chip) but the sizes are very different:

```
VMEM        ~128 MB / chip   (on-chip SRAM — shared by all 4 MXUs + VPU)
HBM         16 GB            ~820 GB/s   (off-chip — same role as GPU HBM)
```

An A100 SM has ~164 KB of shared memory. A TPU v5e has ~128 MB of VMEM. That looks like 800x more on-chip space, but the comparison is slightly misleading: VMEM is chip-wide (shared by all MXUs), while shared memory is per-SM. A fairer comparison might be GPU L2 cache (~40 MB on A100), which is also chip-wide. Still, the TPU has 3x more chip-level on-chip memory, and VMEM is explicitly managed by the compiler rather than being a transparent cache. Bigger tiles fit on-chip, more data reuse per HBM load. Same tiling tradeoff from Part 4 (bigger tiles = more reuse but must fit in SRAM), just with a higher ceiling on TPU.

---

## How data flows through a systolic array {#systolic-array}

I kept reading "systolic array" and thinking I understood it. I did not.

![Systolic array overview — 4x4 array with weights pre-loaded, cell detail, and stagger diagram](/assets/imgs/systolic-array-overview.png)

### Weight-stationary (what the TPU MXU uses)

Weights stay put, everything else flows.

For `C = A @ B` where A is `(M, K)` and B is `(K, N)`:
- The array is K rows x N columns (matching B's dimensions)
- Cell `(k, n)` holds `B[k][n]`, loaded once, never moves
- Activations from A stream in from the left, one element per cell per cycle
- Partial sums flow downward through each column
- Result `C[m][n]` exits from the bottom of column `n`

```
         col 0     col 1
         +-----+   +-----+
A[m,0] > |B[0,0]| > |B[0,1]|   < row 0 (activation passes right)
         +--+--+   +--+--+
            | S       | S        < partial sums flow down
         +--+--+   +--+--+
A[m,1] > |B[1,0]| > |B[1,1]|   < row 1
         +--+--+   +--+--+
            | S       | S
         +--+--+   +--+--+
A[m,2] > |B[2,0]| > |B[2,1]|   < row 2
         +--+--+   +--+--+
            |          |
         C[m,0]     C[m,1]       < results exit bottom
```

Why weight-stationary? In neural network inference, the same weights multiply many different input batches. Loading weights once and streaming activations through means the most expensive data (weights, which are large and reused) never moves.

### The stagger

This is the part I had to stare at. `A[m][k]` doesn't enter row `k` at the same time as `A[m][0]` enters row `0`. It's staggered: `A[m][k]` enters row `k` delayed by `k` cycles. Why? Because partial sums flow downward. Cell `(k, n)` needs to receive both:
1. The activation `A[m][k]` from the left
2. The partial sum from cell `(k-1, n)` above, which takes `k` cycles to get there (flowing down from row 0)

The stagger synchronizes these two data flows. Without it, the activation would arrive before its matching partial sum, or vice versa.

Timing for a (2, 3) @ (3, 2) matmul:

```
Cycle:    0          1          2          3
       +------+  +------+  +------+  +------+
Row 0: |A[0,0]|  |A[1,0]|  |      |  |      |
       +------+  +------+  +------+  +------+
Row 1: |      |  |A[0,1]|  |A[1,1]|  |      |    < delayed by 1
       +------+  +------+  +------+  +------+
Row 2: |      |  |      |  |A[0,2]|  |A[1,2]|    < delayed by 2
       +------+  +------+  +------+  +------+
```

Each new row of A (m=0, m=1) only costs 1 extra cycle. The pipeline is always full, no bubbles between different rows of A within one matmul. Total cycles: `M + K + N - 2`.

### Output-stationary (not the TPU, but it shows up in diagrams)

Searching for systolic array diagrams will often turn up a different design where both A and B stream in, A from the left and B from the top. This is the output-stationary design:

- The array is M rows x N columns (matching C's dimensions)
- Cell `(i, j)` accumulates `C[i][j]`, the result builds up in place
- Both inputs flow through and keep moving

This is the design that shows "both matrices streaming from two sides." It's a valid design, but it's not what the TPU uses. The TPU uses weight-stationary because it minimizes the most expensive data movement for inference workloads.

---

## Building a systolic array emulator {#emulator}

To really understand the timing, I built a tick-based emulator: a `SystolicArray` class with a `tick()` method that advances one cycle, moving data through the pipeline exactly as the hardware would.

```python
class SystolicArray:
    """Fixed-size weight-stationary systolic array emulator (TPU MXU design).

    Dimensions:
        - The array has `num_rows` rows and `num_cols` columns of cells.
        - B (num_rows x num_cols) is pre-loaded into cells — one weight per cell, stationary.
        - A (num_activations x num_rows) streams in from the left, one row of A per cycle,
          staggered: A[m, row] enters at cycle (m + row).
        - Partial sums flow downward through rows. Result C[m, col] exits
          the bottom of column `col` at cycle (m + num_rows + col - 1).
    """

    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows      # K: inner dimension of the matmul
        self.num_cols = num_cols      # N: number of output columns

        self.weights = np.zeros((num_rows, num_cols))
        # NaN means the cell is idle (no activation has arrived yet)
        self.activation_in_cell = np.full((num_rows, num_cols), np.nan)
        # Row 0 starts at 0; each row adds its contribution and passes down
        self.partial_sum = np.zeros((num_rows, num_cols))

        self.cycle = 0
        self._A = None
        self._num_activations = 0
        self._total_cycles = 0
        self._done = False
        self.results = {}             # (m, col) -> final dot product value

    def load_weights(self, B):
        """Pre-load weight matrix B into the array. One weight per cell, stays fixed."""
        assert B.shape == (self.num_rows, self.num_cols)
        self.weights = B.astype(np.float64).copy()

    def start_matmul(self, A):
        """Queue activation matrix A for streaming. Resets all pipeline state."""
        num_activations, inner_dim = A.shape
        assert inner_dim == self.num_rows
        self._A = A.astype(np.float64).copy()
        self._num_activations = num_activations
        self._total_cycles = num_activations + self.num_rows + self.num_cols - 2
        self._done = False
        self.cycle = 0
        self.results = {}
        self.activation_in_cell = np.full((self.num_rows, self.num_cols), np.nan)
        self.partial_sum = np.zeros((self.num_rows, self.num_cols))

    def tick(self):
        """Advance the array by one cycle."""
        current_cycle = self.cycle
        new_activation_in_cell = np.full((self.num_rows, self.num_cols), np.nan)
        new_partial_sum = np.zeros((self.num_rows, self.num_cols))

        for row in range(self.num_rows):
            for col in range(self.num_cols):

                # Step 1: Where does this cell's activation come from?
                if col == 0:
                    # First column: from the input queue.
                    # A[m, row] enters at cycle t = m + row (the stagger).
                    activation_index = current_cycle - row
                    if 0 <= activation_index < self._num_activations:
                        activation = float(self._A[activation_index, row])
                    else:
                        activation = None     # ramp-up or drain phase
                else:
                    # Other columns: passes rightward from the left neighbor.
                    left_neighbor = self.activation_in_cell[row, col - 1]
                    if np.isnan(left_neighbor):
                        activation = None     # left neighbor was idle
                    else:
                        activation = float(left_neighbor)

                # Step 2: Partial sum from above
                if row == 0:
                    incoming_partial_sum = 0.0   # top row starts at zero
                else:
                    incoming_partial_sum = float(self.partial_sum[row - 1, col])

                # Step 3: Compute if we have an activation
                if activation is not None:
                    weight = float(self.weights[row, col])
                    updated_partial_sum = incoming_partial_sum + activation * weight

                    new_activation_in_cell[row, col] = activation
                    new_partial_sum[row, col] = updated_partial_sum

                    # Bottom row: result exits the array
                    if row == self.num_rows - 1:
                        result_index = current_cycle - row - col
                        if 0 <= result_index < self._num_activations:
                            self.results[(result_index, col)] = updated_partial_sum
                else:
                    new_partial_sum[row, col] = incoming_partial_sum

        self.activation_in_cell = new_activation_in_cell
        self.partial_sum = new_partial_sum
        self.cycle += 1
        if self.cycle > self._total_cycles:
            self._done = True

    @property
    def done(self):
        return self._done

    def matmul(self, A, B):
        """Load weights, stream A, tick until done, return result matrix."""
        self.load_weights(B)
        self.start_matmul(A)
        while not self.done:
            self.tick()
        C = np.zeros((self._num_activations, self.num_cols))
        for (m, col), value in self.results.items():
            C[m, col] = value
        return C
```

Quick test:

```
A @ B = [[ 4.  5.]
 [10. 11.]]
Emulator = [[ 4.  5.]
 [10. 11.]]
Match: True
Total cycles: 6  (M+K+N-2+1 = 6)
```

![Cycle-by-cycle systolic array execution — (2,3) @ (3,2) over 6 cycles](/assets/imgs/systolic-array-cycles.png)

The thing I took away from building this: the stagger isn't a complication, it's the mechanism. By delaying `A[m, k]`'s entry into row `k` by exactly `k` cycles, the activation arrives at each cell at the same moment as the matching partial sum from above. The pipeline stays full, no control logic needed.

I wired the emulator into a `TPUCycleSimulator` that counts MXU and VPU cycles for the full attention computation, both flash and standard. For small matrices (all dimensions <= 16), it ticks through the actual systolic array and verifies the cycle count matches the `M + K + N - 2` formula. For larger matrices, it uses the formula directly.

```python
class TPUCycleSimulator:
    """Approximate cycle-level simulation of TPU MXU + VPU.
    Uses the SystolicArray emulator for matmuls — the cycle count
    falls out of the hardware simulation rather than a formula.
    """

    def __init__(self, mxu_dim=128, vpu_width=128):
        self.mxu_dim = mxu_dim
        self.vpu_width = vpu_width
        self.mxu_cycles = 0
        self.vpu_cycles = 0
        self.mxu_flops = 0

    def matmul(self, A, B):
        """Route through the systolic array emulator.
        For tiles that fit (K,N <= 16), tick through actual hardware pipeline.
        The cycle count M+K+N-2 isn't assumed — it's verified.
        """
        M, K = A.shape
        _, N = B.shape
        formula_cycles = M + K + N - 2

        if K <= 16 and N <= 16 and M <= 16:
            arr = SystolicArray(num_rows=K, num_cols=N)
            C = arr.matmul(A, B)
            assert arr.cycle == formula_cycles + 1
        else:
            C = A @ B

        self.mxu_cycles += formula_cycles
        self.mxu_flops += 2 * M * K * N
        return C

    def vpu(self, n_elements, cycles_per_vec=1):
        """Elementwise VPU op. 128 elements per vector.
        Ceiling division: (n-1)//128+1 so exact multiples don't overshoot."""
        self.vpu_cycles += ((n_elements - 1) // self.vpu_width + 1) * cycles_per_vec
```

```
Systolic array cycle counts verified against formula ✓
```

---

## What the emulator revealed {#emulator-revealed}

The simulator compares flash attention (block=128) against standard attention for n=512, d=64:

```
                                    block=64   block=128    standard
  ─────────────────────────────────────────────────────────────────
  Total cycles                        24,556      16,936      20,604
  MXU cycles                          13,680       6,360       2,172
  VPU cycles                          10,876      10,576      18,432
  MXU utilization                       8.4%       20.1%       94.3%
  vs standard                          1.19x       0.82x       1.00x
```

Flash does less total compute for causal attention. It skips entire tiles in the upper triangle, 6 tiles out of 16 for a 4x4 grid. Standard attention processes the full n×n matrix, running `exp(-inf)` on all the masked entries. Flash never touches them at all.

But MXU utilization tells the real story. Even with block=128, flash attention's MXU utilization is only ~20% vs standard's ~94%. Flash has two matmuls per tile: `Q_tile @ K_tile.T` = `(128, 64) @ (64, 128)` and `weights @ V_tile` = `(128, 128) @ (128, 64)`. Both have inner dimension <= d=64 or block=128, so the systolic pipeline runs for at most 128 steps through a 128-wide array. Standard attention's `weights @ V` is `(512, 512) @ (512, 64)`, inner dimension 512, giving the pipeline 512 steps of useful work. That single large matmul is what drives standard's ~94% utilization.

The simulator likely overcounts standard attention though. A fused XLA kernel could recognize the causal mask and skip the upper triangle entirely, never computing `exp(-inf)`, never multiplying by zero weights. The simulator charges full price for the masked entries; a smart compiler probably wouldn't. (Without profiling the actual XLA-generated code, this is speculation, but the benchmark gap is consistent with it.)

So: the algorithm does less compute than standard attention. `vmap` proves it, because once XLA can see the Q-block parallelism, it gets within 2x of the fused path and beats it at large sizes. The remaining gap is likely DMA pipelining and fusion, things only a lower-level API can express. (Dumping the HLO would confirm this; for now it's an educated guess from the benchmark shape.)

### What production code does

`jax.nn.dot_product_attention` is JAX's built-in attention. XLA recognizes the pattern and applies its own optimized implementation:

```python
@jax.jit
def builtin_causal_attention(Q, K, V):
    # Expects (batch..., seq, heads, head_dim) — NOT (seq, d).
    # Add heads=1 dimension: (n, d) -> (n, 1, d) -> call -> squeeze back.
    out = jax.nn.dot_product_attention(
        Q[:, None, :], K[:, None, :], V[:, None, :],
        is_causal=True,
    )
    return out[:, 0, :]
```

The benchmark confirmed it, identical performance to fused standard attention at every size:

```
     n   scores(MB)    VMEM?    standard(ms)   flash(ms)   builtin(ms)   builtin speedup
-------------------------------------------------------------------------------------
   512        1.0      yes         0.070       0.070         0.067            1.05x
  1024        4.0      yes         0.066       0.133         0.079            0.85x
  2048       16.0      yes         0.073       0.521         0.081            0.91x
  4096       64.0      yes         0.073       2.507         0.074            0.99x
  8192      256.0       NO         1.188      14.051         1.189            1.00x
 16384     1024.0       NO         4.444      89.542         4.448            1.00x
 32768     4096.0       NO        17.115     102.995        17.222            0.99x
```

For anything beyond what XLA auto-selects, there's [Splash Attention](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention), Google's TPU-optimized flash attention written in Pallas. It uses DMA pipelining, MXU-matched tile sizes, and 2D grid scheduling, i.e. everything my `fori_loop` couldn't express.

---

## Pallas: what it would take to beat the compiler {#pallas}

So how does Splash Attention actually beat XLA's fused path? Pallas, JAX's equivalent of Triton. You write custom kernels in Python that lower through Mosaic to TPU VLIW instructions.

Three things Pallas provides that pure JAX can't express:

1. DMA pipelining. The `fori_loop` implementation likely does load-wait-compute-load-wait-compute. A Pallas kernel can double-buffer: while the MXU computes on the current tile, the DMA engine fetches the next tile into a separate VMEM buffer. Compute and memory transfer overlap instead of serializing.

2. MXU-matched tiling with causal skipping. A 2D Pallas grid `(num_q_blocks, num_kv_blocks)` gives Mosaic full visibility into the iteration pattern. It knows which tiles are fully masked by the causal triangle and skips them entirely.

3. Explicit VMEM placement. All data movement goes through `BlockSpec` declarations, no dynamic indexing in the kernel body. This is how the hardware knows what to prefetch.

I tried writing one. Mosaic's constraints are restrictive: no dynamic indexing (`k_all[indices, :]` lowers to an unsupported gather), 1D blocks must be multiples of 128, kernels that compile on one JAX version fail on another. The code didn't survive into this post. There's a reason Splash Attention is a [serious engineering effort](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention), not a code snippet.

At this point my brain was pretty thoroughly consumed by the TPU rabbit hole. The Pallas deep dive can wait for another day. In short: use `jax.nn.dot_product_attention` as the default (XLA picks the best strategy), Splash Attention for long sequences at scale where you need kernel-level tuning, and pure JAX `fori_loop` for understanding the algorithm (not for production).

---

## What I actually learned {#what-i-learned}

### The hardware was already doing it

The whole arc of this post: I tried to force a GPU optimization onto a TPU, and for this setup (single head, d=64, Colab v5e), the TPU was already handling it.

Flash attention exists because GPU SRAM is tiny (~164 KB/SM). The n×n score matrix never fits, so tiling in software is mandatory. On TPU, the MXU is literally a tile processor. A 128x128 systolic array that holds one matrix stationary and streams the other through. That's what flash attention implements in software on GPU, but it's what the TPU hardware does by default.

Add ~128 MB of VMEM (800x more on-chip memory than a GPU SM) and XLA's automatic fusion, and the score matrix just stays on-chip. My handwritten tiling was reimplementing what the hardware and compiler already handle, but worse. (At production scale with multi-head, longer sequences, larger d, the tradeoffs shift and Splash Attention becomes necessary. But for the single-head setup I was benchmarking, the compiler had it covered.)

### Giving the compiler information matters more than writing clever code

The 45x speedup from `fori_loop` to `vmap` wasn't a better algorithm. It was the *same* algorithm with one additional piece of information: "these Q blocks are independent." XLA does dataflow analysis, operator fusion, memory planning. But it can't infer independence from a `fori_loop` with carried state. `vmap` is semantically "map this function over a batch," so independence is built into the abstraction.

This is a different skill than writing Triton kernels. In Triton, the programmer is the compiler, deciding what goes where. In JAX, you're having a conversation with a compiler. The better you express intent, the better code it generates. `fori_loop` said "do these sequentially." `vmap` said "these are independent." Same math. 45x difference.

### Tiling is the same idea everywhere, it's just a question of who does it

On TPU, tile-level matmul is in hardware (the MXU is a 128x128 tile), the tiling schedule is the compiler's job (XLA), and on-chip memory management is the compiler's job (VMEM). On GPU, tile-level matmul is in software (tensor cores need warp-level MMA instructions), the tiling schedule is either the programmer's job (Triton/CUDA) or the compiler's (torch.compile), and on-chip memory is managed by the programmer (shared memory).

Same building block: tile, stream, accumulate. TPU pushes more into hardware and compiler. GPU gives more control but requires more work.

### GPU vs TPU, in summary

On the GPU side (Part 4): Triton compiles through LLVM to PTX, I fuse manually (the kernel IS the fusion), tiling is manual pointer arithmetic, I decide what lives in SRAM, there's ~164 KB of SRAM per SM, and flash attention wins because the score matrix never fits on-chip.

On the TPU side (this post): JAX compiles through HLO to XLA to device code, XLA fuses automatically, tiling is implicit (compiler decides) or explicit via `BlockSpec` in Pallas, the compiler decides what lives in VMEM, there's ~128 MB of VMEM per chip, and flash attention only matters once the score matrix exceeds ~n=8K (single head, d=64).

The lesson I keep coming back to: the same optimization has completely different value on different hardware. I spent Parts 3-4 building up flash attention as this essential technique, and it is, on GPU. On TPU, at least for this single-head d=64 setup on a Colab v5e, the hardware architecture makes it unnecessary for typical sequence lengths, and the compiler handles it when it does become necessary. Understanding *why* I lost taught me more about both architectures than winning on GPU did.
