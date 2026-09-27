#!/usr/bin/env python3
"""
Quick smoke test 2: test embedding-based ANN blocking recall
on 500 training entities with embedding model.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE    = Path("/home/harsh/Desktop/CodeNova/Amazon_ML_Challenge/student_resource")
SRC_DIR = BASE / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from normalizer import normalize, get_tokens
from blocking import LexicalBlocker

TRAIN_DIR = BASE / "dataset" / "train"
N_SAMPLE  = 500   # small for speed

print("="*60)
print("SMOKE TEST 2 — Embedding ANN Blocking")
print("="*60)

tr1 = pd.read_csv(TRAIN_DIR/"train_source1.tsv", sep="\t", dtype=str).fillna("")
tr2 = pd.read_csv(TRAIN_DIR/"train_source2.tsv", sep="\t", dtype=str).fillna("")
tr3 = pd.read_csv(TRAIN_DIR/"train_source3.tsv", sep="\t", dtype=str).fillna("")
gt  = pd.read_csv(TRAIN_DIR/"train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
)
gt_dict = {r.source1_entity_id: set(r.match_list) for r in gt.itertuples(index=False)}

from sentence_transformers import SentenceTransformer

print("\nLoading multilingual model...")
t0 = time.time()
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
model.max_seq_length = 128
print(f"  Model loaded in {time.time()-t0:.1f}s")

def make_text(name, addr):
    parts = []
    if name and name.strip(): parts.append(name.strip())
    if addr and addr.strip():  parts.append(addr.strip()[:80])
    return " | ".join(parts) if parts else "unknown"

# Test on India partition (hardest)
s1_india = tr1[tr1['country'] == 'India'].sample(n=N_SAMPLE, random_state=42)
tgt_india = pd.concat([
    tr2[tr2['country'] == 'India'],
    tr3[tr3['country'] == 'India']
], ignore_index=True)
s23_lkp = {r.entity_id: {"name": r.business_name, "addr": r.business_address, "country": r.country}
           for r in tgt_india.itertuples(index=False)}

print(f"\nIndia: S1={len(s1_india)} targets={len(tgt_india):,}")

# Encode targets
t_ids   = list(tgt_india['entity_id'])
t_texts = [make_text(r.business_name, r.business_address) for r in tgt_india.itertuples(index=False)]

print(f"Encoding {len(t_ids):,} targets...")
t0 = time.time()
t_embs = model.encode(t_texts, batch_size=512, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
enc_time = time.time() - t0
enc_speed = len(t_ids) / enc_time
print(f"  Done in {enc_time:.1f}s ({enc_speed:.0f} records/s)")
print(f"  Estimated for full India test (4.7M targets): {4_700_000/enc_speed/60:.0f} min")

# Build FAISS index
import faiss
nlist = min(256, len(t_ids) // 100)
q = faiss.IndexFlatIP(t_embs.shape[1])
idx = faiss.IndexIVFFlat(q, t_embs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
idx.train(t_embs)
idx.nprobe = 32
idx.add(t_embs)
print(f"  FAISS index built: {idx.ntotal} vectors")

# Encode queries
q_texts = [make_text(r.business_name, r.business_address) for r in s1_india.itertuples(index=False)]
q_embs  = model.encode(q_texts, batch_size=512, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

# ANN search
for top_k in [5, 10, 20, 30]:
    scores, indices = idx.search(q_embs, top_k)
    
    found = 0; missed = 0; total_cands = 0
    for i, row in enumerate(s1_india.itertuples(index=False)):
        true_matches = {m for m in gt_dict.get(row.entity_id, set())
                        if m in s23_lkp and s23_lkp[m]['country'] == 'India'}
        cands = {t_ids[j] for j in indices[i] if j >= 0 and scores[i][list(indices[i]).index(j)] >= 0.5}
        total_cands += len(cands)
        for tm in true_matches:
            if tm in cands: found += 1
            else: missed += 1
    
    recall = found / max(found + missed, 1)
    avg_c  = total_cands / max(N_SAMPLE, 1)
    print(f"  top_k={top_k:2d}: Recall={recall*100:.1f}%  found={found} missed={missed}  avg_cands={avg_c:.1f}")

# Show some missed examples at top_k=20
print("\nMissed examples at top_k=20:")
scores20, indices20 = idx.search(q_embs, 20)
count = 0
for i, row in enumerate(s1_india.itertuples(index=False)):
    cands = {t_ids[j] for j in indices20[i] if j >= 0}
    true_matches = {m for m in gt_dict.get(row.entity_id, set())
                    if m in s23_lkp and s23_lkp[m]['country'] == 'India'}
    for tm in true_matches:
        if tm not in cands and count < 5:
            m = s23_lkp[tm]
            print(f"  S1:  '{row.business_name[:55]}'")
            print(f"  S23: '{m['name'][:55]}'")
            print()
            count += 1

print("\n✅ Smoke test 2 complete!")
