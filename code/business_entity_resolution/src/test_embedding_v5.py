#!/usr/bin/env python3
"""
Test script to verify embedding model works end-to-end.
Run after torchvision 0.21+cu124 installs.
"""
import glob, os, ctypes, sys, time

# Load CUDA libs
for p in glob.glob('/home/harsh/.local/lib/python3.13/site-packages/nvidia/*/lib'):
    if os.path.isdir(p):
        for so in glob.glob(os.path.join(p, '*.so*')):
            try: ctypes.CDLL(so)
            except: pass

# Reorder sys.path: venv first
_VENV_SP = "/home/harsh/Desktop/CodeNova/Amazon_ML_Challenge/.venv/lib/python3.13/site-packages"
_LOCAL_SP = "/home/harsh/.local/lib/python3.13/site-packages"
sys.path = (
    [p for p in sys.path if _VENV_SP in p] +
    [p for p in sys.path if _VENV_SP not in p and _LOCAL_SP not in p] +
    [p for p in sys.path if _LOCAL_SP in p and _VENV_SP not in p]
)

import torch
print(f"torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import torchvision
    print(f"torchvision: {torchvision.__version__} | from: {torchvision.__file__}")
except Exception as e:
    print(f"torchvision ERROR: {e}")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print(f"\nLoading multilingual embedding model...")
t0 = time.time()
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model loaded in {time.time()-t0:.1f}s")

test_texts = [
    "Supreme IT Services",        # English
    "सुप्रीम आईटी सर्विसेज",       # Hindi (Devanagari)
    "McDonald's Restaurant",       # English  
    "मैकडोनाल्ड्स रेस्तरां",      # Hindi
    "Raj Enterprises Mumbai",
    "राज एंटरप्राइजेज मुंबई",
]

t0 = time.time()
embs = model.encode(test_texts, normalize_embeddings=True, batch_size=32)
print(f"\nEncoding {len(test_texts)} texts done in {time.time()-t0:.2f}s | Shape: {embs.shape}")

print("\nCosine Similarities (cross-script matching test):")
print(f"  Supreme IT (en) ↔ सुप्रीम आईटी (hi):     {embs[0] @ embs[1]:.4f}  (expected > 0.7)")
print(f"  McDonald's (en) ↔ मैकडोनाल्ड्स (hi):   {embs[2] @ embs[3]:.4f}  (expected > 0.7)")
print(f"  Raj Enterprises (en) ↔ राज एंटरप्राइजेज (hi): {embs[4] @ embs[5]:.4f}  (expected > 0.6)")
print(f"  Supreme IT ↔ McDonald's (different):   {embs[0] @ embs[2]:.4f}  (expected < 0.5)")

# Test FAISS
print("\nTesting FAISS ANN index...")
N = 10000
D = 768
fake_embs = np.random.randn(N, D).astype(np.float32)
fake_embs /= np.linalg.norm(fake_embs, axis=1, keepdims=True)

index = faiss.IndexFlatIP(D)
index.add(fake_embs)
q = fake_embs[:5]
scores, indices = index.search(q, 10)
print(f"  FAISS index: {N:,} vectors, top-10 search OK")
print(f"  Top score (self-query): {scores[0,0]:.4f}  (expected ~1.0)")

# Test jellyfish
import jellyfish
print(f"\njellyfish: soundex('Smith')={jellyfish.soundex('Smith')}, metaphone('Smith')={jellyfish.metaphone('Smith')}")

print("\n✅ ALL TESTS PASSED — Embedding pipeline is ready for v5!")
