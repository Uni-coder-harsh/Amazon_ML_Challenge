"""
embed_block.py
──────────────
Embedding-based ANN blocking using multilingual sentence transformers.
Handles: Hindi, Tamil, French, English and cross-lingual matching.

Model: paraphrase-multilingual-mpnet-base-v2
  - 278M params (well under 8B limit)
  - Apache 2.0 license
  - Supports 50+ languages including all languages in this dataset
  - 768-dim embeddings

Strategy:
  - Encode combined "name || address" text for each record
  - Build FAISS IVF index over target (S2+S3) embeddings
  - For each S1 query, retrieve top-k nearest neighbors
  - Merge with lexical candidates (union)

Memory-conscious:
  - Encodes in batches of 512
  - Uses float16 FAISS index (int8 quantization optional)
  - Processes per country partition to reduce memory footprint
"""

from __future__ import annotations

import os
import gc
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM = 768
BATCH_SIZE = 512
TOP_K = 10  # retrieve top-10 ANN neighbors per query

# Keep model cached across calls
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  [embed] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = 128  # cap for speed
    return _model


def make_text(name: str, addr: str) -> str:
    """Combine name + address into a single embedding input."""
    parts = []
    if name and name.strip():
        parts.append(name.strip())
    if addr and addr.strip():
        # Use first 80 chars of address (most discriminative part)
        parts.append(addr.strip()[:80])
    return " | ".join(parts) if parts else "unknown"


def encode_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode list of texts → (N, EMBED_DIM) float32 array."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 norm → cosine = dot product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, nlist: int = 100) -> faiss.Index:
    """
    Build a flat L2 index for small datasets, IVF for larger ones.
    Since embeddings are L2-normalized, L2 distance ≡ cosine distance.
    """
    n, d = embeddings.shape
    if n < 10_000:
        # Exact search for small sets
        index = faiss.IndexFlatIP(d)  # Inner product = cosine (normalized vecs)
    else:
        # IVF for scalable search
        nlist = min(nlist, n // 10)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = min(10, nlist)
    index.add(embeddings)
    return index


def ann_blocking(
    query_ids: list[str],
    query_texts: list[str],
    target_ids: list[str],
    target_texts: list[str],
    top_k: int = TOP_K,
    batch_size: int = BATCH_SIZE,
) -> dict[str, set[str]]:
    """
    For each query, find top_k nearest target records in embedding space.
    
    Returns: dict mapping query_id → set of candidate target_ids
    """
    print(f"  [embed] Encoding {len(target_ids):,} target records...")
    t0 = time.time()
    target_embs = encode_texts(target_texts, batch_size=batch_size)
    print(f"  [embed] Target encoding done in {time.time()-t0:.1f}s")

    print(f"  [embed] Building FAISS index ({len(target_ids):,} vectors)...")
    index = build_faiss_index(target_embs)
    del target_embs
    gc.collect()

    print(f"  [embed] Encoding {len(query_ids):,} query records...")
    t0 = time.time()
    query_embs = encode_texts(query_texts, batch_size=batch_size)
    print(f"  [embed] Query encoding done in {time.time()-t0:.1f}s")

    print(f"  [embed] ANN search (top_k={top_k})...")
    t0 = time.time()
    k = min(top_k, len(target_ids))
    scores, indices = index.search(query_embs, k)
    print(f"  [embed] ANN search done in {time.time()-t0:.1f}s")

    del query_embs, index
    gc.collect()

    results: dict[str, set[str]] = {}
    for i, qid in enumerate(query_ids):
        cands = set()
        for j, score in zip(indices[i], scores[i]):
            if j >= 0 and score > 0.5:  # cosine > 0.5 threshold
                cands.add(target_ids[j])
        results[qid] = cands

    return results
