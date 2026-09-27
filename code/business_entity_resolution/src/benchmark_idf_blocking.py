import pandas as pd
import numpy as np
import re
import unicodedata
import time
from collections import defaultdict, Counter

TRAIN_DIR = "dataset/train"
print("Loading India data...")
t0 = time.time()
tr1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str).fillna("")
tr2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str).fillna("")
tr3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str).fillna("")
gt  = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
)
gt_dict = {r.source1_entity_id: set(r.match_list) for r in gt.itertuples(index=False)}

# Filter India
s1_india = tr1[tr1['country'] == 'India']
tgt_india = pd.concat([
    tr2[tr2['country'] == 'India'],
    tr3[tr3['country'] == 'India']
], ignore_index=True)
print(f"Loaded in {time.time()-t0:.1f}s: S1_India={len(s1_india):,}, Targets_India={len(tgt_india):,}")

# Preprocessing helpers
LEGAL_RE = re.compile(r"\b(pvt\.?|private|ltd\.?|limited|llc|inc\.?|corp\.?|corporation|co\.?|llp|lp|gmbh|ag|sas|sarl|sa)\b", re.I)
EXT_RE = re.compile(r"(\.com|\.in|\.org|\.net|\.io|@|#)", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9\s]")

GENERIC_NAME_STOP = {
    'enterprises', 'enterprise', 'services', 'service', 'solutions', 'solution',
    'group', 'company', 'india', 'international', 'holdings', 'holding',
    'industries', 'industry', 'consultancy', 'consultants', 'associates'
}
GENERIC_ADDR_STOP = {
    'the', 'and', 'near', 'opp', 'opposite', 'road', 'street', 'floor', 'flat',
    'house', 'plot', 'shop', 'building', 'bldg', 'lane', 'nagar', 'city', 'state',
    'dist', 'district', 'area', 'colony', 'complex', 'block', 'sector', 'sec', 'flr'
}

def normalize_latin(s: str) -> str:
    # De-accent
    s = unicodedata.normalize('NFKD', s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()

def get_clean_compact_name(s: str) -> str:
    s = normalize_latin(s)
    s = LEGAL_RE.sub(" ", s)
    s = EXT_RE.sub(" ", s)
    return re.sub(r"[^a-z0-9]", "", s)

def get_name_tokens(s: str) -> set:
    s = normalize_latin(s)
    s = LEGAL_RE.sub(" ", s)
    s = EXT_RE.sub(" ", s)
    s = NON_ALNUM.sub(" ", s)
    tokens = set()
    for w in s.split():
        if len(w) >= 3 and w not in GENERIC_NAME_STOP:
            tokens.add(w)
    return tokens

def get_addr_numbers(s: str) -> set:
    # Numbers normalized by stripping leading zeros
    nums = set()
    for n in re.findall(r"\d+", s):
        clean_n = n.lstrip('0')
        if clean_n and len(clean_n) >= 2: # at least 2 digits to avoid noise like 1st floor
            nums.add(clean_n)
    return nums

def get_addr_tokens(s: str) -> set:
    s = normalize_latin(s)
    s = NON_ALNUM.sub(" ", s)
    tokens = set()
    for w in s.split():
        if len(w) >= 3 and w not in GENERIC_ADDR_STOP and not w.isdigit():
            tokens.add(w)
    return tokens

# Build Inverted Indexes on ALL 4.1M India Targets
print("\nBuilding inverted indexes on 4.1M targets...")
t0 = time.time()
idx_compact_name = defaultdict(list)
idx_name_token   = defaultdict(list)
idx_addr_num     = defaultdict(list)
idx_addr_token   = defaultdict(list)

# We will also compute document frequency (DF) to weight tokens!
df_name_token = Counter()
df_addr_token = Counter()
df_addr_num   = Counter()

for row in tgt_india.itertuples(index=False):
    eid = row.entity_id
    nm = row.business_name
    ad = row.business_address
    
    # 1. Compact name
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4:
        idx_compact_name[cname].append(eid)
    
    # 2. Name tokens
    ntoks = get_name_tokens(nm)
    for t in ntoks:
        idx_name_token[t].append(eid)
        df_name_token[t] += 1
        
    # 3. Addr numbers
    anums = get_addr_numbers(ad)
    for n in anums:
        idx_addr_num[n].append(eid)
        df_addr_num[n] += 1
        
    # 4. Addr tokens
    atoks = get_addr_tokens(ad)
    for t in atoks:
        idx_addr_token[t].append(eid)
        df_addr_token[t] += 1

print(f"Indexes built in {time.time()-t0:.1f}s.")
print(f"Unique compact names: {len(idx_compact_name):,}")
print(f"Unique name tokens: {len(idx_name_token):,}")
print(f"Unique address numbers: {len(idx_addr_num):,}")
print(f"Unique address tokens: {len(idx_addr_token):,}")

# Test on 2,000 random S1 queries that have true matches
sample_s1 = s1_india[s1_india['entity_id'].map(lambda x: len(gt_dict.get(x, set())) > 0)].sample(2000, random_state=42)
print(f"\nQuerying {len(sample_s1)} entities against 4.1M target index...")

N_TARGETS = len(tgt_india)
import math

# IDF helper: rare tokens give high score, common tokens give low score
def get_idf(df, N):
    return math.log((N + 1) / (df + 1)) + 1.0

# Query function
def query_entity(row, top_k=20):
    nm = row.business_name
    ad = row.business_address
    
    candidate_scores = defaultdict(float)
    
    # 1. Exact compact name match (massive signal!)
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4:
        if cname in idx_compact_name:
            for cid in idx_compact_name[cname]:
                candidate_scores[cid] += 50.0
                
    # 2. Name tokens (weighted by IDF)
    ntoks = get_name_tokens(nm)
    for t in ntoks:
        df = df_name_token.get(t, 0)
        if 0 < df < 10000: # skip hyper-frequent words
            idf = get_idf(df, N_TARGETS)
            for cid in idx_name_token[t]:
                candidate_scores[cid] += idf * 2.0
                
    # 3. Addr numbers (weighted by IDF)
    anums = get_addr_numbers(ad)
    for n in anums:
        df = df_addr_num.get(n, 0)
        if 0 < df < 15000: # skip common numbers like 10, 11
            idf = get_idf(df, N_TARGETS)
            for cid in idx_addr_num[n]:
                candidate_scores[cid] += idf * 2.5
                
    # 4. Addr tokens (weighted by IDF)
    atoks = get_addr_tokens(ad)
    for t in atoks:
        df = df_addr_token.get(t, 0)
        if 0 < df < 20000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_addr_token[t]:
                candidate_scores[cid] += idf * 1.0
                
    if not candidate_scores:
        return []
        
    # Return top_k candidates by score
    top = sorted(candidate_scores.items(), key=lambda x: -x[1])[:top_k]
    return top

t0 = time.time()
recalls = {5: 0, 10: 0, 15: 0, 20: 0, 30: 0}
total_true_matches = 0
cand_counts = []

for row in sample_s1.itertuples(index=False):
    true_matches = gt_dict.get(row.entity_id, set())
    total_true_matches += len(true_matches)
    
    ranked = query_entity(row, top_k=30)
    cand_counts.append(len(ranked))
    
    cand_ids = [cid for cid, s in ranked]
    for k in [5, 10, 15, 20, 30]:
        top_k_ids = set(cand_ids[:k])
        recalls[k] += len(true_matches & top_k_ids)

q_time = time.time() - t0
print(f"2000 queries completed in {q_time:.1f}s ({len(sample_s1)/q_time:.1f} queries/sec)!")
print(f"Average candidates returned (cap=30): {np.mean(cand_counts):.1f}")
for k in [5, 10, 15, 20, 30]:
    r = recalls[k] / total_true_matches
    print(f"  Recall @ Top-{k:2d}: {recalls[k]}/{total_true_matches} = {r*100:.2f}%")
