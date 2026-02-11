---
layout: post
title:  "How Attention Actually Works"
date:   2026-02-11 12:00:00 -0400
category: professional
math: true
---

I gave a talk to my team about how LLMs work under the hood. This is that talk turned into a blog post. There are three parts planned. This first one is about attention.

I'll use math and code you can run. You can paste all of this into a Python REPL and run it.

---

**Contents**
- [What is an LLM?](#what-is-an-llm)
- [Attention in one line](#attention-in-one-line)
- [Why scale by $\sqrt{d_k}$?](#why-scale-by-sqrt-d_k)
- [Causal masking](#causal-masking)
- [Multi-head attention](#multi-head-attention)
- [Embeddings are not fixed labels](#embeddings-are-not-fixed-labels)

---

## What is an LLM?

It is the same block repeated dozens of times. Each block refines the token representations a little more:

```
"I love cats" -> [tokenize + embed]
                       |
              each token is a vector: (seq_len, d_model)
                       |
              +---- Layer 1 ------+
              |  Attention        |  <- tokens look at each other
              |  Feed-Forward     |  <- each token thinks on its own
              +-------------------+
                       |            still (seq_len, d_model) -- same shape!
              +---- Layer 2 ------+
              |  Attention        |
              |  Feed-Forward     |
              +-------------------+
                       |
                   ... x80 ...
                       |
              +---- Layer 80 -----+
              |  Attention        |
              |  Feed-Forward     |
              +-------------------+
                       |
              (seq_len, d_model)
                       |
              Take LAST position -> (1, d_model)
                       |
              x unembedding matrix  (d_model, vocab_size)
                       |
              -> logits (1, vocab_size)  e.g. (1, 128000)
                       |
              softmax -> P(next token)
              "are": 61%, "is": 12%, "were": 8%, ...
```

The unembedding matrix is often just the transpose of the embedding matrix. Same weights, opposite direction. One matmul, not an MLP.

We are going to focus on the **Attention** part. The rest (FFN, normalization, residual connections) matters but is simpler.

---

## Attention in one line

```
Attention(Q, K, V) = softmax(Q @ K.T / √d_k) @ V
```

Each token looks at every other token it is allowed to see, figures out how relevant they are, and takes a weighted mix of their values.

### Toy example: 3 tokens, 2 dimensions

Tokens start as embeddings. In real models `d_model` is often a few thousand. We use 2 dimensions here so you can see all the numbers:

```
X = ┌ "I"    → [ 1.0,  0.5 ] ┐
    │ "love" → [ 0.8,  0.2 ] │   shape: (3 tokens, 2 dims)
    └ "cats" → [ 0.3,  0.9 ] ┘
```

The model projects `X` through three learned weight matrices:

```
Q = X @ W_Q   "what am I looking for?"       (3, 2)
K = X @ W_K   "what do I contain?"            (3, 2)
V = X @ W_V   "what info do I carry?"         (3, 2)
```

For this toy example, pretend `Q ≈ K ≈ V ≈ X` (no projection). I'm also skipping the `√d_k` scale here for readability; it doesn't change the ranking for small `d_k`. Here is what happens step by step:

**Step 1: `Q @ K.T` — how similar is each token to every other token?**
```
Scores = Q @ K.T        (3,2) @ (2,3) → (3,3)

                  K: "I"   "love"  "cats"
        Q: "I"   [ 1.25   0.90    0.75 ]
        "love"   [ 0.90   0.68    0.42 ]
        "cats"   [ 0.75   0.42    0.90 ]
```

**Step 2: `softmax` — turn each row into a probability distribution**
```
Weights = softmax(Scores)     each row sums to 1.0

                  K: "I"   "love"  "cats"
        Q: "I"   [ 0.43   0.30    0.26 ]
        "love"   [ 0.41   0.33    0.26 ]
        "cats"   [ 0.35   0.25    0.40 ]
```

**Step 3: `Weights @ V` — mix values according to those weights**
```
Output = Weights @ V     (3,3) @ (3,2) → (3,2)

        "I"    → [ 0.75,  0.51 ]     was [1.0, 0.5] — now blended
        "love" → [ 0.74,  0.49 ]     was [0.8, 0.2] — shifted toward "I"
        "cats" → [ 0.67,  0.57 ]     was [0.3, 0.9] — now carries info from "I" and "love"
```

Before attention, `"cats"` only knew it was `"cats"`. After attention, it has blended in information from `"I"` and `"love"`. It now knows something about being cats-that-are-loved-by-I. Do this 80 times and the representations get very detailed.

### Q, K, V — what are they?

| Matrix | What it means | Analogy |
|--------|-----------|--------|
| **Query (Q)** | "What am I looking for?" | A search query |
| **Key (K)** | "What do I contain?" | Index / metadata |
| **Value (V)** | "What do I give if picked?" | The actual content |

Concrete example. In "The cat sat on the mat because **it** was tired":
- `Q` for `"it"` encodes something like "I need to find what I refer to"
- `K` for `"cat"` encodes something like "I am an animate singular noun"
- `K` for `"mat"` encodes something like "I am an inanimate noun"
- The dot product `Q·K` is high between `"it"` and `"cat"`, so attention pulls `"cat"`'s information into `"it"`

### The code

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights
```

Let's run it with random data and trace through each step:

```python
seq_len, d_k = 4, 3
np.random.seed(42)
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

scores = Q @ K.T
print("Step 1 — Q @ K.T (attention scores):")
print(scores)
```
```
[[-0.732  0.064  0.694 -1.044]
 [ 1.22  -0.693 -1.395 -0.026]
 [-0.276 -1.813 -3.206  0.364]
 [ 1.821  0.018 -0.521  0.51 ]]
```

```python
weights = softmax(scores)
print("Step 2 — softmax (each row sums to 1):")
print(weights)
```
```
[[0.123 0.273 0.513 0.09 ]
 [0.663 0.098 0.048 0.191]
 [0.316 0.068 0.017 0.599]
 [0.653 0.108 0.063 0.176]]
```

```python
output = weights @ V
print("Step 3 — weights @ V:")
print(output)
print(f"Shape: {output.shape} — same as input")
```
```
[[-0.369  0.874 -0.339]
 [-0.555  0.261 -1.025]
 [-0.79   0.518 -1.115]
 [-0.539  0.269 -0.999]]
Shape: (4, 3) — same as input
```

---

## Why scale by $\sqrt{d_k}$? {#why-scale-by-sqrt-d_k}

Most explanations skip this part. Let's do it.

When `d_k` is large, dot products get large too. Large scores make softmax saturate: one token gets almost all the mass, gradients vanish.

We can see it directly:

```python
def show_variance(d_k, n_samples=10000):
    q = np.random.randn(n_samples, d_k)
    k = np.random.randn(n_samples, d_k)
    dots = np.sum(q * k, axis=1)
    print(f"d_k={d_k:>3}  std={dots.std():.1f}  (expected: {np.sqrt(d_k):.1f})")

show_variance(3)
show_variance(64)
show_variance(512)
```
```
d_k=  3  std=1.7  (expected: 1.7)
d_k= 64  std=7.9  (expected: 8.0)
d_k=512  std=22.3  (expected: 22.6)
```

Standard deviation grows with $\sqrt{d_k}$. Why?

<details>
<summary>Proof: why Var[q · k] = d_k</summary>
<div markdown="1">

Each entry of Q and K has mean 0, variance 1. The dot product is a sum of $d_k$ terms:

$$\mathbf{q} \cdot \mathbf{k} = q_1 k_1 + q_2 k_2 + \ldots + q_{d_k} k_{d_k}$$

Each term $q_i k_i$ has variance 1:

$$\text{Var}[q_i k_i] = E[q_i^2] \cdot E[k_i^2] = 1 \cdot 1 = 1$$

(by independence, since $E[q_i] = 0$).

The terms are independent, so $\text{Var}[\mathbf{q} \cdot \mathbf{k}] = d_k$. Standard deviation is $\sqrt{d_k}$. Divide by $\sqrt{d_k}$ and variance goes back to 1.

</div>
</details>

That keeps softmax in a useful range:

```python
print(softmax(np.array([1., 2., 3.])))
print(softmax(np.array([20., -20., 0.])))
```
```
[0.09  0.245 0.665]    ← spread out, useful gradients
[1.    0.    0.   ]    ← saturated, dead gradients
```

---

## Causal masking

When generating text, token `t` can only look at tokens `0` through `t-1`. If it could see the future it would be cheating.

We add `-∞` to the upper triangle of the score matrix before softmax. `exp(-∞) = 0`, so those positions get zero weight:

```python
def causal_attention(Q, K, V):
    d_k = Q.shape[-1]
    seq_len = Q.shape[0]
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    scores = scores + mask
    weights = softmax(scores)
    return weights @ V, weights

_, weights = causal_attention(Q, K, V)
print(weights)
```
```
[[1.    0.    0.    0.   ]
 [0.751 0.249 0.    0.   ]
 [0.627 0.258 0.115 0.   ]
 [0.481 0.17  0.124 0.225]]
```

Token 0 can only see itself. Token 1 sees tokens 0-1. Token 3 sees everything before it. Each row still sums to 1.

---

## Multi-head attention

Each head tends to specialize in a kind of pattern. But language has many things going on at once.

Take "The cat sat on the mat because **it** was comfortable." To figure out what "it" means, you need to track:
- Syntax: "it" is the subject of "was"
- Reference: "it" points to "mat" (or "cat"?)
- Meaning: "comfortable" describes what exactly?

So we run multiple attention heads in parallel. Each one has its own Q, K, V weights and learns a different pattern:

```
Head 0: maybe learns "look at the previous noun"
Head 1: maybe learns "look at the verb I go with"
Head 2: maybe learns "look at things with similar meaning"
...
```

We split d_model into h heads, each of size d_k = d_model / h. Run attention on each head, concatenate, project back:

```python
def multihead_attention(X, W_Q, W_K, W_V, W_O, num_heads, causal=True):
    seq_len, d_model = X.shape
    d_k = d_model // num_heads

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    def split_heads(x):
        return x.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if causal:
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    head_outputs = weights @ V

    concat = head_outputs.transpose(1, 0, 2).reshape(seq_len, d_model)
    return concat @ W_O, weights
```

```python
d_model, num_heads, seq_len = 64, 8, 16
np.random.seed(42)
X = np.random.randn(seq_len, d_model).astype(np.float32)
W_Q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_K = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_V = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_O = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

output, attn_weights = multihead_attention(X, W_Q, W_K, W_V, W_O, num_heads)
print(f"Input:  {X.shape}")
print(f"Output: {output.shape}")
print(f"Attention weights: {attn_weights.shape}")
print(f"  = ({num_heads} heads, {seq_len} queries, {seq_len} keys)")
```
```
Input:  (16, 64)
Output: (16, 64)
Attention weights: (8, 16, 16)
  = (8 heads, 16 queries, 16 keys)
```

---

## Embeddings are not fixed labels

This is the thing that ties it all together.

The embedding layer is a lookup table. `"cat"` maps to some vector `[0.3, 0.9, ...]`. A positional encoding is added so the model knows _where_ each token is in the sequence (we are skipping the details here — just know it exists). After that, every `"cat"` at the same position starts with the same vector.

But after attention mixes in the surrounding tokens, it is not just `"cat"` anymore:

```
"My fixed male yellow cat sat on the mat"

Layer 0:   cat → [just the word "cat"]
Layer 10:  cat → [cat + male + yellow + fixed]
Layer 40:  cat → [cat + male + yellow + fixed + My + sitting + on(mat)]
Layer 80:  cat → [everything the model knows about this specific cat
                  in this specific sentence]
```

A 4096-dimensional vector has room for a lot of properties at once. Each attention layer blends in more context. By layer 80, the vector for "cat" has been completely shaped by its surroundings. That is why the same word behaves differently in different sentences.

At this point it isn't really a word anymore. It is a dense encoding of meaning-in-context.

---

## Recap

1. **Attention** = Q @ K.T → softmax → @ V. Each token gets a weighted mix of other tokens' values.
2. **√d_k scaling** keeps dot products from blowing up softmax.
3. **Causal mask** blocks the future. That is what makes generation possible.
4. **Multi-head** runs several attention patterns in parallel.
5. **Embeddings change** layer by layer. By the end they encode full context, not just the word.

Next: how LLMs generate text one token at a time, and why long contexts are expensive.
