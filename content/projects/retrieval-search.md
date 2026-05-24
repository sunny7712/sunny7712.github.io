## The problem

Keyword search over a 20,000-item product catalog was missing too many obvious matches. Users typed natural language; the catalog spoke SKU.

## What I did

Embedded the catalog with gte-base-1.5, indexed in FAISS, and added a text-to-image path with PyTorch-on-GPU. Wrapped the whole thing in Streamlit so I could ship it before losing interest.

## What it taught me

Vector search isn't magic, but the gap between BM25 and a half-decent embedding model is bigger than I expected. Also: Streamlit is a wildly good prototyping tool.
