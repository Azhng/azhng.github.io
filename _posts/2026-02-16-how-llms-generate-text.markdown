---
layout: post
title:  "How LLMs Generate Text"
date:   2026-02-16 12:00:00 -0500
category: professional
math: true
---

This is Part 2. [Part 1 covered attention](/how-attention-actually-works) — how tokens look at each other and share information. This post covers how that machinery generates text one token at a time, and why long contexts are expensive.

Like Part 1, this post includes math and runnable Python. Most examples use numpy; the capstone uses llama-cpp-python.

---

**Contents**
- [The generation loop](#the-generation-loop)
- [Once generated, tokens are fixed](#once-generated-tokens-are-fixed)
- [What each step actually costs](#what-each-step-actually-costs)
- [KV caching](#kv-caching)
- [Where your VRAM goes](#where-your-vram-goes)
- [Prefill vs decode](#prefill-vs-decode)
- [Capstone: a real transformer](#capstone-a-real-transformer)
- [Why this matters in practice](#why-this-matters-in-practice)

---

## The generation loop

Any sequence of tokens has a joint probability. The chain rule of probability lets us factor it:

$$P(x_1, x_2, \ldots, x_n) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1,x_2) \cdot \ldots \cdot P(x_n|x_1,\ldots,x_{n-1})$$

Each term is a conditional: given everything before, what comes next? An LLM is trained to model these conditionals. That's the whole game — predict the next token given everything before it.

Generation is a loop:
1. Feed all tokens so far through the model
2. Get a probability distribution over the vocabulary
3. Pick a token
4. Append it
5. Repeat

Here is a toy version. A real transformer uses attention over all previous tokens. This one only looks at the last token (a bigram model). Both produce the same type of output: **a probability distribution over the entire vocabulary.**

```python
import numpy as np

vocab = ["the", "cat", "dog", "sat", "ran", "on", "mat", "and", ".", "!"]

def dist(high_probs):
    """Full distribution over vocab. Remaining mass spread over unlisted tokens."""
    remaining = 1.0 - sum(high_probs.values())
    unlisted = [t for t in vocab if t not in high_probs]
    full = dict(high_probs)
    for t in unlisted:
        full[t] = remaining / len(unlisted)
    return full

# Our tiny model: P(next token | current token)
# Each entry is a full distribution over the ENTIRE vocabulary.
tiny_model = {
    "the":  dist({"cat": 0.35, "dog": 0.30, "mat": 0.25, "and": 0.08}),
    "cat":  dist({"sat": 0.50, "ran": 0.30, "on": 0.10, ".": 0.08}),
    "dog":  dist({"ran": 0.45, "sat": 0.30, "on": 0.15, ".": 0.08}),
    "sat":  dist({"on": 0.60, ".": 0.20, "and": 0.18}),
    "on":   dist({"the": 0.70, "mat": 0.20, "and": 0.08}),
    "mat":  dist({".": 0.50, "and": 0.30, "!": 0.10, "the": 0.08}),
    "ran":  dist({".": 0.40, "and": 0.30, "on": 0.20, "the": 0.08}),
    "and":  dist({"the": 0.50, "cat": 0.15, "dog": 0.15, "mat": 0.10}),
    ".":    dist({"the": 0.60, "and": 0.38}),
    "!":    dist({"the": 0.60, "and": 0.38}),
}
```

What does the model output? A probability over all 10 tokens — even unlikely ones get some mass:

```python
print(f"P(· | 'the') — distribution over all {len(vocab)} tokens:\n")
for t, p in sorted(tiny_model["the"].items(), key=lambda x: -x[1]):
    print(f"  {t:>5}: {p:>5.1%}")
```
```
P(· | 'the') — distribution over all 10 tokens:

    cat:  35.0%
    dog:  30.0%
    mat:  25.0%
    and:   8.0%
    sat:   0.3%
    ran:   0.3%
     on:   0.3%
      .:   0.3%
      !:   0.3%
    the:   0.3%
```

A real model does the same thing, but over ~50,000 tokens. Here is the generation loop:

{% raw %}
```python
token = "the"
sequence = [token]

for step in range(6):
    probs = tiny_model[token]              # O(1) dict lookup. A real transformer? Full attention over all tokens so far.
    next_token = max(probs, key=probs.get)  # greedy: pick highest probability
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    top3_str = ", ".join(f"{t}: {p:.0%}" for t, p in top3)
    print(f'  Step {step}: P(· | "{token}") = {{{top3_str}, ...}}')
    print(f'           -> "{next_token}"')
    token = next_token
    sequence.append(token)

print(f'\n  Result: {" ".join(sequence)}')
```
{% endraw %}
```
  Step 0: P(· | "the") = {cat: 35%, dog: 30%, mat: 25%, ...}
           -> "cat"
  Step 1: P(· | "cat") = {sat: 50%, ran: 30%, on: 10%, ...}
           -> "sat"
  Step 2: P(· | "sat") = {on: 60%, .: 20%, and: 18%, ...}
           -> "on"
  Step 3: P(· | "on") = {the: 70%, mat: 20%, and: 8%, ...}
           -> "the"
  Step 4: P(· | "the") = {cat: 35%, dog: 30%, mat: 25%, ...}
           -> "cat"
  Step 5: P(· | "cat") = {sat: 50%, ran: 30%, on: 10%, ...}
           -> "sat"

  Result: the cat sat on the cat sat
```

Greedy decoding picks the most likely token every time. Notice the output loops: "the cat sat on" repeats. We'll fix that with [temperature sampling](#temperature) later. For now, the point is the loop: predict, pick, append, repeat.

---

## Once generated, tokens are fixed

This detail matters.

When the model generates token 50, it is conditioned on tokens 0 through 49. That includes its own previous outputs. It cannot go back and change token 10. Token 10 is part of the context now, just like the user's original prompt.

This is why LLM coding tools can get stuck in a loop:
1. Early tokens commit to an approach ("Let's use recursion...")
2. All later tokens are conditioned on that commitment
3. If the approach is wrong, the model keeps building on it
4. It is not being stubborn. It literally cannot revise what it already wrote

The fix is mechanical: start a new context. The old commitment is gone, so the model is free to try something else.

---

## What each step actually costs

The bigram model above does a dict lookup — O(1). A real transformer does something much more expensive.

At each generation step, the model feeds **all tokens so far** through causal attention. At step 2, that means computing a 3×3 attention matrix. At step 5, a 6×6. Every step recomputes the full matrix from scratch.

Trace the same "the cat sat on the cat" sequence, now with transformer attention:

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Same sequence the toy model generated
tokens = ["the", "cat", "sat", "on", "the", "cat", "sat"]

# Random embeddings and projection weights (a real model learns these)
np.random.seed(42)
d_model, d_k = 8, 4
token_embeddings = {t: np.random.randn(d_model) for t in set(tokens)}
W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)

print("What a transformer computes at each step:\n")
for step in range(len(tokens) - 1):
    context = tokens[:step+1]
    X = np.array([token_embeddings[tok] for tok in context])  # (step+1, d_model)
    Q = X @ W_Q                           # queries for each token
    K = X @ W_K                           # keys for each token
    scores = Q @ K.T / np.sqrt(d_k)       # (step+1) x (step+1) attention matrix
    mask = np.triu(np.ones_like(scores) * -np.inf, k=1)
    weights = softmax(scores + mask)

    n = step + 1
    waste = f"  (recomputed rows 0–{step-1} for nothing)" if step > 0 else ""
    print(f'  Step {step}: "{" ".join(context)}" → {n}x{n} attention{waste}')
    print(f'           used last row → predict "{tokens[step+1]}"')
```
```
What a transformer computes at each step:

  Step 0: "the" → 1x1 attention
           used last row → predict "cat"
  Step 1: "the cat" → 2x2 attention  (recomputed rows 0–0 for nothing)
           used last row → predict "sat"
  Step 2: "the cat sat" → 3x3 attention  (recomputed rows 0–1 for nothing)
           used last row → predict "on"
  Step 3: "the cat sat on" → 4x4 attention  (recomputed rows 0–2 for nothing)
           used last row → predict "the"
  Step 4: "the cat sat on the" → 5x5 attention  (recomputed rows 0–3 for nothing)
           used last row → predict "cat"
  Step 5: "the cat sat on the cat" → 6x6 attention  (recomputed rows 0–4 for nothing)
           used last row → predict "sat"
```

At step 5, we compute a 6×6 matrix but only need the last row. Rows 0 through 4 were already computed at previous steps — identically, because the causal mask means no earlier token ever looks at a later one. We computed them and threw them away.

That is the waste KV caching eliminates.

---

## KV caching

In the causal attention matrix from Part 1, adding a new token does not change earlier rows. The causal mask guarantees this: no earlier token attends to a future one.

So we can cache the K and V vectors from previous steps. To generate the next token, we compute **one new row** of attention instead of redoing the whole matrix:

![KV Cache: each step only computes one new row](/assets/imgs/kv-cache-steps.png)

Each step, only the orange row is new. The blue rows are cached — identical to what we computed at earlier steps. The numbers confirm it: row values don't change as the matrix grows.

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

seq_len, d_model, d_k = 8, 8, 4
np.random.seed(42)
X_all = np.random.randn(seq_len, d_model)   # (seq_len, d_model) — token embeddings
W_Q = np.random.randn(d_model, d_k)         # (d_model, d_k) — learned projection weights
W_K = np.random.randn(d_model, d_k)         # (d_model, d_k)

# WITHOUT cache: project ALL tokens and recompute full score matrix at every step
print("Without KV cache:")
for step in range(seq_len):
    X = X_all[:step+1]                             # (step+1, d_model)
    Q = X @ W_Q                                    # (step+1, d_k)
    K = X @ W_K                                    # (step+1, d_k)
    scores = Q @ K.T / np.sqrt(d_k)               # (step+1, step+1) — each entry is one dot product
    print(f"  Step {step}: {step+1}x{step+1} score matrix = {(step+1)**2} dot products")
total_naive = sum((s+1)**2 for s in range(seq_len))
print(f"  Total: {total_naive}")
```
```
Without KV cache:
  Step 0: 1x1 score matrix = 1 dot products
  Step 1: 2x2 score matrix = 4 dot products
  Step 2: 3x3 score matrix = 9 dot products
  Step 3: 4x4 score matrix = 16 dot products
  Step 4: 5x5 score matrix = 25 dot products
  Step 5: 6x6 score matrix = 36 dot products
  Step 6: 7x7 score matrix = 49 dot products
  Step 7: 8x8 score matrix = 64 dot products
  Total: 204
```

Each dot product is $Q_i \cdot K_j$ — a $d_k$-dimensional vector dot product ($d_k$ multiplies + $d_k - 1$ additions). We count dot products rather than scalar operations because $d_k$ is constant across all steps — it does not change the ratio between cached and uncached.

```python
# WITH cache: project each new token ONCE, cache the result
# (We only show the K cache here. V is cached the same way — used after softmax to compute the output.)
print("With KV cache:")
k_cache = []
for step in range(seq_len):
    k_new = X_all[step:step+1] @ W_K    # (1, d_model) @ (d_model, d_k) → (1, d_k)
    k_cache.append(k_new[0])            # store the (d_k,) vector — this is what "KV cache" means
    q_new = X_all[step:step+1] @ W_Q    # (1, d_k)
    K_cached = np.array(k_cache)         # (step+1, d_k) — all cached keys so far
    scores = q_new @ K_cached.T / np.sqrt(d_k)  # (1, d_k) @ (d_k, step+1) → (1, step+1) — one row!
    print(f"  Step {step}: 1x{step+1} score row = {step+1} dot products")
total_cached = sum(s+1 for s in range(seq_len))
print(f"  Total: {total_cached}")
print(f"\n  {total_naive} vs {total_cached} = {total_naive/total_cached:.1f}x less work")
```
```
With KV cache:
  Step 0: 1x1 score row = 1 dot products
  Step 1: 1x2 score row = 2 dot products
  Step 2: 1x3 score row = 3 dot products
  Step 3: 1x4 score row = 4 dot products
  Step 4: 1x5 score row = 5 dot products
  Step 5: 1x6 score row = 6 dot products
  Step 6: 1x7 score row = 7 dot products
  Step 7: 1x8 score row = 8 dot products
  Total: 36

  204 vs 36 = 5.7x less work
```

With caching, generating token `t` costs O(t). Total for n tokens:

$$1 + 2 + 3 + \cdots + n = \frac{n(n+1)}{2} = O(n^2)$$

Without caching, each step recomputes the full matrix: O(t²) at step t, O(n³) total. The cache drops it from cubic to quadratic.

| | Without cache | With cache |
|---|---|---|
| Cost at step t | O(t²) full matrix | O(t) one new row |
| Memory | O(1) nothing stored | O(n) cache grows |
| Total for n tokens | O(n³) | O(n²) |

The tradeoff: more memory (storing all those K, V vectors) in exchange for less redundant computation.

---

## Where your VRAM goes

The KV cache stores K and V for every token, at every layer. The formula:

```
KV cache per token = 2 (K and V) × num_layers × num_kv_heads × head_dim × bytes_per_param
```

`bytes_per_param` is 2 for BF16/FP16 (the standard serving precision) — each value is stored as a 16-bit float. Quantized serving (e.g. INT8) cuts this to 1 byte.

In classic multi-head attention (MHA), every attention head has its own K and V, so `num_kv_heads × head_dim = d_model`. Modern models often use **Grouped Query Attention (GQA)**[^gqa]: several query heads share one K/V head, which reduces KV memory. [Llama 3](https://ai.meta.com/blog/meta-llama-3/) uses 8 KV heads across model sizes:

```python
models = {
    "Llama 3 8B":    {"layers": 32,  "kv_heads": 8, "head_dim": 128, "bytes": 2},
    "Llama 3 70B":   {"layers": 80,  "kv_heads": 8, "head_dim": 128, "bytes": 2},
    "Llama 3.1 405B": {"layers": 126, "kv_heads": 8, "head_dim": 128, "bytes": 2},
}

print(f"{'Model':<18} {'KV heads':>8} {'Per token':>10} {'@ 4K ctx':>9} {'@ 128K ctx':>11}")
print("-" * 60)
for name, m in models.items():
    per_token = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * m["bytes"]
    at_4k = 4096 * per_token / 1e9
    at_128k = 131072 * per_token / 1e9
    print(f"{name:<18} {m['kv_heads']:>8} {per_token/1e6:>8.2f} MB {at_4k:>8.1f} GB {at_128k:>9.1f} GB")
```
```
Model              KV heads  Per token   @ 4K ctx  @ 128K ctx
------------------------------------------------------------
Llama 3 8B                8   0.13 MB      0.5 GB     17.2 GB
Llama 3 70B               8   0.33 MB      1.3 GB     43.0 GB
Llama 3.1 405B            8   0.52 MB      2.1 GB     67.6 GB
```

This is **on top of model weights.** An 8B model in fp16 takes ~16 GB. A 70B model takes ~140 GB. The KV cache is additional.

GQA keeps per-token KV cost low. But it still adds up:
- **Long contexts are expensive.** At 128K tokens, even the 8B model's cache is 17 GB — a meaningful fraction of an A100's 80 GB.
- **Serving many users multiplies it.** Each user gets their own KV cache. Ten concurrent users at 4K context on a 70B model: ~13 GB just for caches. Servers use *continuous batching* to interleave requests, so many caches are alive at once.
- **Context window limits are partly a memory constraint** — but also depend on the position encoding and how the model was trained.
- **Quantization helps.** Serving in INT8 instead of BF16 halves the KV cache.

---

## Prefill vs decode

When you use an LLM, there is usually a pause before text starts streaming.

1. **Prefill** (the pause): process the entire prompt at once. Populate the KV cache for all layers. Cost: O(S²) where S is prompt length.
2. **Decode** (the stream): generate one token at a time, each one adding a row to the KV cache. Cost: O(S+t) per token.

```
You paste 50K tokens of code + a question

[=== prefill (several seconds) ===][token][token][token][token]...
^                                   ^
slow: processing your whole prompt  fast: one token at a time
```

The pause is called **Time to First Token (TTFT)**. Longer prompt means longer wait. And because attention is O(S²), the cost scales quadratically — double the prompt, quadruple the wait:

```python
base = 8  # 8K as baseline
for n in [8, 16, 32, 64, 128, 200]:
    cost = (n / base) ** 2
    bar = "#" * max(1, int(cost / 16))
    print(f"  {n:>4}K context: {cost:>7.1f}x  {bar}")
```
```
     8K context:     1.0x  #
    16K context:     4.0x  #
    32K context:    16.0x  #
    64K context:    64.0x  ####
   128K context:   256.0x  ################
   200K context:   625.0x  #######################################
```

128K is 256 times more expensive than 8K.

---

## Capstone: a real transformer

Everything above used numpy and toy models. Next, run the same generation loop on a real transformer — [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), 135 million parameters, running locally via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

We use the low-level API so we can see every step: eval, get logits, softmax, sample, eval again.

```python
from llama_cpp import Llama
import numpy as np

llm = Llama(model_path="SmolLM2-135M-Instruct-Q8_0.gguf",
            n_ctx=512, logits_all=True, verbose=False)
# SmolLM2-135M: 30 layers, 9 heads, 3 KV heads (GQA), head_dim=64, vocab=49152

prompt_tokens = llm.tokenize(b"The cat sat on")
# prompt_tokens = [504, 2644, 2643, 335]  →  ["The", " cat", " sat", " on"]

# --- PREFILL: feed the entire prompt in one call ---
llm.reset()                   # clear KV cache
llm.eval(prompt_tokens)       # forward pass on all 4 prompt tokens
# KV cache now holds K, V for all 4 tokens, at all 30 layers.
# That's 4 × 30 × 3 × 64 × 2 (K+V) × 2 bytes = 90 KB.

# --- DECODE: generate one token at a time ---
max_tokens = 15
generated = []

for step in range(max_tokens):
    # 1. Get logits for the last position — (vocab_size,) = (49152,)
    logits = llm._scores[llm.n_tokens - 1].copy()

    # 2. Softmax → probability distribution over entire vocabulary
    probs = softmax(logits)

    # 3. Greedy: pick the highest-probability token
    #    argmax returns an index into the vocab — that index IS the token ID
    token_id = int(np.argmax(probs))
    token_str = llm.detokenize([token_id]).decode(errors="replace")
    generated.append(token_str)

    # Show top 5 candidates — same view as our toy model
    top5 = np.argsort(probs)[::-1][:5]
    candidates = "  ".join(
        f"{llm.detokenize([t]).decode(errors='replace').strip()}: {probs[t]*100:.0f}%"
        for t in top5
    )
    print(f"  Step {step:>2}: {candidates}")
    print(f"           -> '{token_str.strip()}'")

    # 4. Feed JUST the new token back — this is KV caching in action.
    #    The cache already has K, V for all previous tokens.
    #    eval([token_id]) computes one new K, V pair and appends to the cache.
    #    Without caching, we'd need: eval(prompt_tokens + all_generated_so_far)
    llm.eval([token_id])

print(f"\n  Result: The cat sat on{''.join(generated)}")
# KV cache: 19 entries (4 prompt + 15 generated). Each eval([token_id]) added one row.
```
```
  Step  0: the: 65%  my: 9%  a: 8%  her: 3%  top: 3%
           -> 'the'
  Step  1: porch: 9%  window: 8%  edge: 6%  bed: 3%  windows: 3%
           -> 'porch'
  Step  2: ,: 60%  and: 4%  swing: 4%  of: 3%  .: 3%
           -> ','
  Step  3: watching: 48%  its: 5%  staring: 5%  her: 4%  looking: 2%
           -> 'watching'
  Step  4: the: 83%  me: 2%  you: 2%  over: 2%  her: 2%
           -> 'the'
  Step  5: sun: 17%  stars: 12%  world: 7%  sunset: 6%  birds: 5%
           -> 'sun'
  Step  6: set: 33%  rise: 26%  go: 9%  dip: 6%  come: 4%
           -> 'set'
  Step  7: behind: 49%  over: 27%  .: 7%  ,: 5%  below: 4%
           -> 'behind'
  Step  8: the: 74%  it: 9%  a: 9%  her: 2%  him: 1%
           -> 'the'
  Step  9: trees: 63%  mountains: 5%  hills: 2%  house: 1%  rolling: 1%
           -> 'trees'
  Step 10: .: 68%  ,: 22%  and: 5%  like: 1%  as: 1%
           -> '.'
  Step 11: It: 16%  She: 15%  The: 15%  I: 12%  : 7%
           -> 'It'
  Step 12: was: 83%  had: 3%  seemed: 3%  felt: 1%  's: 1%
           -> 'was'
  Step 13: a: 55%  peaceful: 10%  the: 5%  beautiful: 2%  like: 2%
           -> 'a'
  Step 14: peaceful: 39%  quiet: 15%  beautiful: 5%  serene: 5%  perfect: 4%
           -> 'peaceful'

  Result: The cat sat on the porch, watching the sun set behind the trees. It was a peaceful
```

### Temperature

Greedy decoding always picks the top token — deterministic but repetitive (remember the toy model looping "the cat sat on the cat sat"?). **Temperature** scales the logits before softmax: `softmax(logits / temp)`.

- Lower temp → sharper distribution → more predictable text
- Higher temp → flatter distribution → more creative (or incoherent)
- `temp → 0` collapses to greedy

```python
def generate(llm, prompt, max_tokens=20, temp=0.0):
    tokens = llm.tokenize(prompt.encode())
    llm.reset()
    llm.eval(tokens)
    generated = []
    for step in range(max_tokens):
        logits = llm._scores[llm.n_tokens - 1].copy()
        if temp < 0.01:
            token_id = int(np.argmax(logits))          # greedy
        else:
            probs = softmax(logits / temp)              # temperature scaling
            token_id = int(np.random.choice(len(probs), p=probs))
        generated.append(token_id)
        llm.eval([token_id])
    return llm.detokenize(generated).decode(errors="replace")

for temp in [0.0, 0.3, 0.8, 1.5]:
    text = generate(llm, "The cat sat on", max_tokens=20, temp=temp)
    label = "greedy" if temp == 0.0 else f"{temp}"
    print(f"  temp={label:<6}  The cat sat on{text}")
```
```
  temp=greedy  The cat sat on the porch, watching the sun set behind the trees. It was a peaceful moment, one that reminded
  temp=0.3     The cat sat on the porch, watching the sun set behind the trees. I sat at my desk, sipping my coffee
  temp=0.8     The cat sat on the bed, staring blankly at the blank sheet to its left, and why not, the blank
  temp=1.5     The cat sat on 5 rose bushes growing this directions out through mourbing gait continuing scenario (Once down many poppy flowers
```

At temp=0.3, the output is close to greedy. At temp=1.5, it often becomes incoherent. There's also **top-k** (only consider the k most likely tokens) and **top-p** / nucleus sampling (consider the smallest set of tokens whose cumulative probability exceeds p), but the generation loop is the same.

Compare this to the toy bigram loop at the top:

| | Toy bigram | Real transformer |
|---|---|---|
| Get distribution | `probs = tiny_model[token]` | `llm.eval([token_id])` then `softmax(llm._scores[-1])` |
| Vocab size | 10 tokens | 49,152 tokens |
| Pick next | `max(probs, key=probs.get)` | `np.argmax(probs)` |
| The loop | identical | identical |

The difference is where the distribution comes from. The loop structure is the same.

---

## Why this matters in practice

| You see | The mechanism | What helps |
|---------|--------------|------------|
| Keeps making the same mistake | Autoregressive: early wrong tokens lock in the approach, everything after builds on them | Fresh context, explicit "stop and reconsider" |
| Seems to forget instructions | Long context dilutes attention — models retrieve better from the [beginning and end](https://arxiv.org/abs/2307.03172) than from the middle | Repeat key instructions, put them at start and end |
| Makes up file contents | Predicts what would plausibly come next. No file in context means it generates plausible-looking content | Make sure files are actually in context |
| Slow on long prompts | O(n²) prefill. 100K context is ~156x more expensive than 8K | Use the minimum context you need |

The key point: **context is not a bucket you dump things into.** It is input to an attention mechanism with real costs and limits. Use it deliberately.

---

## Recap

1. **Generation** = predict next token, pick one, append, repeat. Always forward, never backward.
2. **Tokens are permanent.** Once generated, they are part of the context. The model cannot revise them.
3. **KV caching** stores previous K, V vectors. Drops generation from O(n³) to O(n²).
4. **VRAM** = model weights + KV cache per user. GQA helps, but long contexts still eat memory.
5. **Prefill vs decode.** The pause before streaming is O(n²) prefill. Double context, quadruple the wait.
6. **Same loop, real model.** SmolLM2-135M runs the identical predict-pick-append loop. Temperature controls randomness; top-k and top-p control which tokens are even considered.


[^gqa]: GQA groups multiple query heads under a single K/V head. DeepSeek's [Multi-Latent Attention (MLA)](https://arxiv.org/abs/2405.04434) compresses KV further into a low-rank latent vector; MLA is out of scope for this post.
