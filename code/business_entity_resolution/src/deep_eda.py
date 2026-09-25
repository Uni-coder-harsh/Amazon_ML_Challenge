#!/usr/bin/env python3
"""
Deep EDA Script for Amazon ML Challenge - Business Entity Resolution
Analyzes both training and test data to understand:
1. Script/language distribution (Latin vs Indic vs other)
2. Noise patterns (name/address variations)
3. Blocking recall analysis on training data
4. Ground truth structure analysis
5. Token overlap between matched pairs
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from collections import Counter, defaultdict
import json
import os
import sys

BASE = "/home/harsh/Desktop/CodeNova/Amazon_ML_Challenge/student_resource"
TRAIN_DIR = f"{BASE}/dataset/train"
TEST_DIR = f"{BASE}/dataset/test"

print("="*70)
print("DEEP EDA - Amazon ML Challenge: Business Entity Resolution")
print("="*70)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────
print("\n[1] Loading datasets...")
tr1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str).fillna("")
tr2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str).fillna("")
tr3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str).fillna("")
gt  = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

te1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t", dtype=str).fillna("")
te2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t", dtype=str).fillna("")
te3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t", dtype=str).fillna("")

print(f"  Train S1: {len(tr1):,} | S2: {len(tr2):,} | S3: {len(tr3):,}")
print(f"  Test  S1: {len(te1):,} | S2: {len(te2):,} | S3: {len(te3):,}")
print(f"  Ground truth rows: {len(gt):,}")

# ── 2. COUNTRY DISTRIBUTION ────────────────────────────────────────────────
print("\n[2] Country Distribution:")
for name, df in [("Train-S1", tr1), ("Train-S2", tr2), ("Train-S3", tr3),
                  ("Test-S1",  te1), ("Test-S2",  te2), ("Test-S3",  te3)]:
    cvc = df['country'].value_counts()
    print(f"  {name}: {dict(cvc)}")

# ── 3. SCRIPT DETECTION ────────────────────────────────────────────────────
def detect_script(text: str) -> str:
    """Detect dominant script in text."""
    if not text or not text.strip():
        return "EMPTY"
    counts = Counter()
    for ch in text:
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "UNKNOWN")
        if "LATIN" in name:
            counts["LATIN"] += 1
        elif "DEVANAGARI" in name:
            counts["DEVANAGARI"] += 1
        elif "TAMIL" in name:
            counts["TAMIL"] += 1
        elif "ARABIC" in name:
            counts["ARABIC"] += 1
        elif "TELUGU" in name:
            counts["TELUGU"] += 1
        elif "KANNADA" in name:
            counts["KANNADA"] += 1
        elif "MALAYALAM" in name:
            counts["MALAYALAM"] += 1
        elif "BENGALI" in name:
            counts["BENGALI"] += 1
        elif "GUJARATI" in name:
            counts["GUJARATI"] += 1
        elif "GURMUKHI" in name:
            counts["GURMUKHI"] += 1
        elif cat.startswith("L"):
            counts["OTHER_LETTER"] += 1
        elif cat.startswith("N"):
            counts["NUMBER"] += 1
        else:
            counts["PUNCT/SPACE"] += 1
    
    letters = {k: v for k, v in counts.items() 
               if k not in ("NUMBER", "PUNCT/SPACE")}
    if not letters:
        return "NO_LETTERS"
    dom = max(letters, key=letters.get)
    return dom

print("\n[3] Script Distribution in Business Names:")
for name, df in [("Train-S1", tr1), ("Train-S2", tr2), ("Train-S3", tr3),
                  ("Test-S1",  te1), ("Test-S2",  te2), ("Test-S3",  te3)]:
    # Sample 50k for speed
    sample = df['business_name'].dropna().sample(min(50000, len(df)), random_state=42)
    scripts = Counter(sample.apply(detect_script))
    total = len(sample)
    print(f"  {name} (n={total:,}):")
    for sc, cnt in scripts.most_common():
        print(f"    {sc}: {cnt:,} ({cnt/total*100:.1f}%)")

# ── 4. EMPTY / MISSING VALUE ANALYSIS ─────────────────────────────────────
print("\n[4] Missing/Empty Value Analysis:")
for name, df in [("Train-S1", tr1), ("Train-S2", tr2), ("Train-S3", tr3),
                  ("Test-S1",  te1), ("Test-S2",  te2), ("Test-S3",  te3)]:
    n = len(df)
    n_empty_name = (df['business_name'].str.strip() == "").sum()
    n_empty_addr = (df['business_address'].str.strip() == "").sum()
    print(f"  {name}: empty_name={n_empty_name:,} ({n_empty_name/n*100:.1f}%) | "
          f"empty_addr={n_empty_addr:,} ({n_empty_addr/n*100:.1f}%)")

# ── 5. GROUND TRUTH ANALYSIS ───────────────────────────────────────────────
print("\n[5] Ground Truth Structure:")
gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
)
gt['n_matches'] = gt['match_list'].apply(len)

print(f"  Total S1 entities: {len(gt):,}")
print(f"  Singletons (no matches): {(gt['n_matches']==0).sum():,} ({(gt['n_matches']==0).mean()*100:.1f}%)")
print(f"  With ≥1 match: {(gt['n_matches']>0).sum():,} ({(gt['n_matches']>0).mean()*100:.1f}%)")
print(f"  Match count distribution:")
for k, v in sorted(Counter(gt['n_matches']).items()):
    bar = "█" * min(int(v/5000), 40)
    print(f"    {k:2d} matches: {v:7,} {bar}")

# S2 vs S3 match breakdown
s2_ids = set(tr2['entity_id'])
s3_ids = set(tr3['entity_id'])
all_matched = [m for lst in gt['match_list'] for m in lst]
n_s2 = sum(1 for m in all_matched if m.startswith("S2-"))
n_s3 = sum(1 for m in all_matched if m.startswith("S3-"))
print(f"\n  S2 matches: {n_s2:,} | S3 matches: {n_s3:,}")
print(f"  S2 coverage: {n_s2/len(tr2)*100:.1f}% of S2 records are matched")
print(f"  S3 coverage: {n_s3/len(tr3)*100:.1f}% of S3 records are matched")

# ── 6. BLOCKING RECALL ANALYSIS ────────────────────────────────────────────
print("\n[6] Blocking Recall Analysis (on training data):")

def normalize_name(text: str) -> str:
    """Simple NFKD normalization for Latin text only."""
    if not text:
        return ""
    try:
        t = unicodedata.normalize("NFKD", text)
        t = t.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Remove legal suffixes
    suffixes = r"\b(pvt|private|ltd|limited|llc|inc|corp|corporation|co|llp|lp|gmbh|ag|sas|sarl|sa)\b"
    t = re.sub(suffixes, "", t)
    return re.sub(r"\s+", " ", t).strip()

def get_tokens(text: str) -> set:
    return set(t for t in normalize_name(text).split() if len(t) >= 3)

# Sample 10k S1 entities with matches for analysis
sample_gt = gt[gt['n_matches'] > 0].sample(min(10000, (gt['n_matches']>0).sum()), random_state=42)

# Build lookup dicts
s1_dict = tr1.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
s2_dict = tr2.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
s3_dict = tr3.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
s23_dict = {**s2_dict, **s3_dict}

zero_token_overlap = 0
total_pairs = 0
token_overlap_dist = []
latin_mismatch = 0  # S1 is Latin but S2/S3 is not (or vice versa)
addr_only_recoverable = 0
completely_lost = 0

name_sim_scores = []
addr_sim_scores = []

from rapidfuzz import fuzz

for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_dict:
        continue
    s1 = s1_dict[s1_id]
    s1_tokens = get_tokens(s1['business_name'])
    s1_addr_tokens = get_tokens(s1['business_address'])
    s1_script = detect_script(s1['business_name'])
    
    for m_id in row['match_list']:
        if m_id not in s23_dict:
            continue
        m = s23_dict[m_id]
        m_tokens = get_tokens(m['business_name'])
        m_script = detect_script(m['business_name'])
        m_addr_tokens = get_tokens(m['business_address'])
        
        total_pairs += 1
        overlap = len(s1_tokens & m_tokens)
        token_overlap_dist.append(overlap)
        
        if overlap == 0:
            zero_token_overlap += 1
            # Can we recover via address?
            addr_overlap = len(s1_addr_tokens & m_addr_tokens)
            if addr_overlap > 0:
                addr_only_recoverable += 1
            else:
                completely_lost += 1
        
        if s1_script != m_script and s1_script not in ("EMPTY", "NO_LETTERS") and m_script not in ("EMPTY", "NO_LETTERS"):
            latin_mismatch += 1
        
        # Name similarity
        ns = fuzz.token_sort_ratio(s1['business_name'], m['business_name']) / 100.0
        name_sim_scores.append(ns)
        
        # Address similarity
        if s1['business_address'] and m['business_address']:
            as_ = fuzz.token_sort_ratio(s1['business_address'], m['business_address']) / 100.0
            addr_sim_scores.append(as_)

print(f"  Analyzed {total_pairs:,} true pairs from {len(sample_gt):,} S1 entities")
print(f"  Zero name-token overlap: {zero_token_overlap:,} ({zero_token_overlap/total_pairs*100:.1f}%) -- BLOCKING FAILURES")
print(f"    → Recoverable via address tokens: {addr_only_recoverable:,} ({addr_only_recoverable/total_pairs*100:.1f}%)")
print(f"    → Completely lost (no overlap anywhere): {completely_lost:,} ({completely_lost/total_pairs*100:.1f}%)")
print(f"  Script mismatch (Latin↔non-Latin): {latin_mismatch:,} ({latin_mismatch/total_pairs*100:.1f}%)")

print(f"\n  Name fuzzy similarity (token_sort_ratio):")
ns = np.array(name_sim_scores)
print(f"    mean={ns.mean():.3f} median={np.median(ns):.3f} p25={np.percentile(ns,25):.3f} p75={np.percentile(ns,75):.3f}")
print(f"    <0.3: {(ns<0.3).sum():,} ({(ns<0.3).mean()*100:.1f}%)")
print(f"    0.3-0.7: {((ns>=0.3)&(ns<0.7)).sum():,} ({((ns>=0.3)&(ns<0.7)).mean()*100:.1f}%)")
print(f"    ≥0.7: {(ns>=0.7).sum():,} ({(ns>=0.7).mean()*100:.1f}%)")

if addr_sim_scores:
    as_ = np.array(addr_sim_scores)
    print(f"\n  Address fuzzy similarity (token_sort_ratio):")
    print(f"    mean={as_.mean():.3f} median={np.median(as_):.3f} p25={np.percentile(as_,25):.3f} p75={np.percentile(as_,75):.3f}")

print(f"\n  Token overlap distribution:")
toc = Counter(token_overlap_dist)
for k in sorted(toc.keys())[:10]:
    bar = "█" * min(int(toc[k]/200), 30)
    print(f"    overlap={k}: {toc[k]:,} {bar}")

# ── 7. EXAMPLES OF HARD CASES ─────────────────────────────────────────────
print("\n[7] Hard Case Examples (zero name token overlap):")
count = 0
for _, row in sample_gt.iterrows():
    if count >= 10:
        break
    s1_id = row['source1_entity_id']
    if s1_id not in s1_dict:
        continue
    s1 = s1_dict[s1_id]
    s1_tokens = get_tokens(s1['business_name'])
    
    for m_id in row['match_list']:
        if m_id not in s23_dict:
            continue
        m = s23_dict[m_id]
        m_tokens = get_tokens(m['business_name'])
        
        if len(s1_tokens & m_tokens) == 0 and s1['business_name'] and m['business_name']:
            print(f"  S1: '{s1['business_name'][:60]}' (country={s1['country']})")
            print(f"  Match: '{m['business_name'][:60]}' (country={m['country']})")
            print(f"  S1 addr: '{s1['business_address'][:60]}'")
            print(f"  Match addr: '{m['business_address'][:60]}'")
            print()
            count += 1
            break

# ── 8. CURRENT PIPELINE ANALYSIS ──────────────────────────────────────────
print("\n[8] Analyzing Current Pipeline Output:")
output_dir = f"{BASE}/output1"
if os.path.exists(f"{output_dir}/matching_results.tsv"):
    results = pd.read_csv(f"{output_dir}/matching_results.tsv", sep="\t", dtype=str).fillna("")
    results['match_list'] = results['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    results['n_matches'] = results['match_list'].apply(len)
    print(f"  Total S1 entities in output: {len(results):,}")
    print(f"  Singletons predicted: {(results['n_matches']==0).sum():,} ({(results['n_matches']==0).mean()*100:.1f}%)")
    print(f"  With ≥1 match: {(results['n_matches']>0).sum():,} ({(results['n_matches']>0).mean()*100:.1f}%)")
    match_dist = Counter(results['n_matches'])
    print(f"  Predicted match count distribution (top 10):")
    for k in sorted(match_dist.keys())[:10]:
        print(f"    {k:2d}: {match_dist[k]:,}")
else:
    print("  No output found")

# ── 9. NAME LENGTH / COMPLEXITY ANALYSIS ──────────────────────────────────
print("\n[9] Name Length & Complexity:")
for name, df in [("Train-S1", tr1), ("Train-S2", tr2), ("Train-S3", tr3),
                  ("Test-S1",  te1), ("Test-S2",  te2), ("Test-S3",  te3)]:
    lens = df['business_name'].str.len()
    wds = df['business_name'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    print(f"  {name}: char_len mean={lens.mean():.1f} median={lens.median():.1f} | "
          f"word_count mean={wds.mean():.1f} median={wds.median():.1f}")

# ── 10. TRAIN vs TEST DISTRIBUTION ────────────────────────────────────────
print("\n[10] Train vs Test - Business Name First Character Distribution:")
for name, df in [("Train-S2", tr2), ("Test-S2", te2)]:
    # First char category
    first_chars = df['business_name'].str[:1].dropna()
    latin_pct = first_chars.apply(lambda c: "LATIN" in unicodedata.name(c, "") if c else False).mean()
    print(f"  {name}: Latin first char = {latin_pct*100:.1f}%")

print("\n[EDA Complete]")
