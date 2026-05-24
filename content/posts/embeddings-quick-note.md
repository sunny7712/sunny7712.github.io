Almost every retrieval system I've worked on uses one of two similarity metrics: dot product $a \cdot b$ or cosine similarity. People sometimes treat them as different things; mostly they aren't.

## The two definitions

Dot product is the obvious one. For two vectors $a, b \in \mathbb{R}^n$:

$$a \cdot b = \sum_{i=1}^{n} a_i b_i$$

Cosine similarity normalizes by magnitude:

$$\cos(\theta) = \frac{a \cdot b}{\|a\| \, \|b\|}$$

If $\|a\| = \|b\| = 1$ — i.e., your embeddings are L2-normalized — then $\cos(\theta) = a \cdot b$. They are the same number.

## When the distinction matters

If your embedding model produces unit-norm vectors (most modern sentence encoders do), reach for raw dot product. It's one less square root per query and FAISS will thank you.

If your vectors have unbounded magnitude — counts, term frequencies, anything you built by hand — use cosine. Otherwise, a long document with high term counts looks "more similar" to everything than a short one, which is rarely what you want.

## What this looks like in practice

For our product-search system at $20{,}000$ items, L2-normalizing once at index time and then using inner product as the FAISS metric gave us a ~$15\%$ latency improvement over computing cosine on the fly. Same recall, fewer cycles. The kind of micro-optimization that's only worth it when the normalization step has nothing to do with the rest of your hot path — which, conveniently, it doesn't.
