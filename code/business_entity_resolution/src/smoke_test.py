#!/usr/bin/env python3
"""
Quick smoke test on 5000 train entities to validate the full pipeline
before running the expensive full training run.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE    = Path("/home/harsh/Desktop/CodeNova/Amazon_ML_Challenge/student_resource")
SRC_DIR = BASE / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from normalizer import normalize, get_tokens, get_ngrams, extract_digits
from blocking import LexicalBlocker
from features import compute_features, N_FEATURES

TRAIN_DIR = BASE / "dataset" / "train"
N_SAMPLE  = 5000

print("="*60)
print("SMOKE TEST — 5000 training entities")
print("="*60)

# Load
tr1 = pd.read_csv(TRAIN_DIR/"train_source1.tsv", sep="\t", dtype=str).fillna("")
tr2 = pd.read_csv(TRAIN_DIR/"train_source2.tsv", sep="\t", dtype=str).fillna("")
tr3 = pd.read_csv(TRAIN_DIR/"train_source3.tsv", sep="\t", dtype=str).fillna("")
gt  = pd.read_csv(TRAIN_DIR/"train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
)
gt_dict = {r.source1_entity_id: set(r.match_list) for r in gt.itertuples(index=False)}

# Sample India partition (hardest due to Indic scripts)
s1_india = tr1[tr1['country'] == 'India'].sample(n=min(N_SAMPLE//2, 2500), random_state=42)
s1_us    = tr1[tr1['country'] == 'US'].sample(n=min(N_SAMPLE//2, 2500), random_state=42)
s1_sample = pd.concat([s1_india, s1_us], ignore_index=True)

tgt_df = pd.concat([tr2, tr3], ignore_index=True)
s23_lkp = {r.entity_id: {"name": r.business_name, "addr": r.business_address, "country": r.country}
           for r in tgt_df.itertuples(index=False)}
s1_lkp  = {r.entity_id: {"name": r.business_name, "addr": r.business_address, "country": r.country}
           for r in tr1.itertuples(index=False)}

# Test LEXICAL BLOCKING per country
total_true_pairs = 0
total_found = 0
total_cands = 0

for country in ['India', 'US']:
    s1_part  = s1_sample[s1_sample['country'] == country]
    tgt_part = tgt_df[tgt_df['country'] == country]

    print(f"\n[Lexical Blocking: {country}] S1={len(s1_part)} targets={len(tgt_part):,}")
    t0 = time.time()

    blocker = LexicalBlocker(max_candidates=50)
    blocker.index([(r.entity_id, r.business_name, r.business_address)
                   for r in tgt_part.itertuples(index=False)])

    found = 0; missed = 0; n_cands = 0
    for r in s1_part.itertuples(index=False):
        true_matches = gt_dict.get(r.entity_id, set())
        cands = blocker.query(r.business_name, r.business_address, use_ngrams=True)
        n_cands += len(cands)
        for tm in true_matches:
            if tm in s23_lkp and s23_lkp[tm]['country'] == country:
                total_true_pairs += 1
                if tm in cands:
                    found += 1
                    total_found += 1
                else:
                    missed += 1

    total_cands += n_cands
    recall = found / max(found + missed, 1)
    avg_c  = n_cands / max(len(s1_part), 1)
    print(f"  Recall={recall*100:.1f}%  found={found} missed={missed}  avg_cands={avg_c:.1f}  ({time.time()-t0:.1f}s)")

    # Show some missed examples
    if missed > 0:
        print("  Sample missed pairs:")
        count = 0
        for r in s1_part.itertuples(index=False):
            cands = blocker.query(r.business_name, r.business_address, use_ngrams=True)
            for tm in gt_dict.get(r.entity_id, set()):
                if tm in s23_lkp and s23_lkp[tm]['country'] == country and tm not in cands:
                    m = s23_lkp[tm]
                    print(f"    S1: '{r.business_name[:50]}' | S2/3: '{m['name'][:50]}'")
                    count += 1
                    if count >= 5: break
            if count >= 5: break

print(f"\n[OVERALL LEXICAL] Recall={total_found/max(total_true_pairs,1)*100:.1f}%  avg_cands={total_cands/max(N_SAMPLE,1):.1f}")

# Test feature computation
print("\n[Feature Test]")
test_pairs = [
    ("Supreme It Private Limited", "Office No S 07 82Haware", "सुप्रीम आईटी प्राइवेट लिमिटेड", "Offiec No S ##07 82Haware"),
    ("Family Empire Partners LLC", "2716 Valdez Drive, Temple TX", "familyempirepartners.com", "2716 VALDEZ DRIVE, TEMPLE, TX"),
    ("Cornerstone Investments LLC", "1475 Manitowoc Road", "@cornerstoneinvestments", "1475 MANITOAOC RD"),
    ("Totally Different Name Inc", "123 Main St", "ABC Corporation", "456 Oak Ave"),
]
for na, aa, nb, ab in test_pairs:
    feat = compute_features(na, aa, nb, ab, country_a="India", country_b="India")
    print(f"  '{na[:30]}' vs '{nb[:30]}'")
    print(f"    name_sort={feat[0]:.2f} partial={feat[1]:.2f} ngram={feat[8]:.2f} addr_sort={feat[4]:.2f}")

print("\n✅ Smoke test complete!")
print("Run pipeline: cd student_resource && ../.venv/bin/python code/business_entity_resolution/src/run_pipeline.py")
