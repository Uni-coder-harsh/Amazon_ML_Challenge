import pandas as pd
import numpy as np
import re
import unicodedata
import math
from collections import defaultdict, Counter
import time

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

s1_india = tr1[tr1['country'] == 'India']
tgt_india = pd.concat([
    tr2[tr2['country'] == 'India'],
    tr3[tr3['country'] == 'India']
], ignore_index=True)
print(f"Loaded in {time.time()-t0:.1f}s: S1_India={len(s1_india):,}, Targets_India={len(tgt_india):,}")

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
    'dist', 'district', 'area', 'colony', 'complex', 'block', 'sector', 'sec', 'flr', 'rd', 'st'
}

def normalize_latin(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()

def get_clean_compact_name(s: str) -> str:
    s = normalize_latin(s)
    s = LEGAL_RE.sub(" ", s)
    s = EXT_RE.sub(" ", s)
    return re.sub(r"[^a-z0-9]", "", s)

def get_name_tokens(s: str) -> list:
    s = normalize_latin(s)
    s = LEGAL_RE.sub(" ", s)
    s = EXT_RE.sub(" ", s)
    s = NON_ALNUM.sub(" ", s)
    tokens = []
    for w in s.split():
        if len(w) >= 3 and w not in GENERIC_NAME_STOP:
            tokens.append(w)
    return tokens

def get_addr_numbers(s: str) -> list:
    nums = []
    for n in re.findall(r"\d+", s):
        clean_n = n.lstrip('0')
        if clean_n:
            nums.append(clean_n)
    return nums

def get_addr_tokens(s: str) -> list:
    s = normalize_latin(s)
    s = NON_ALNUM.sub(" ", s)
    tokens = []
    for w in s.split():
        if len(w) >= 3 and w not in GENERIC_ADDR_STOP and not w.isdigit():
            tokens.append(w)
    return tokens

print("\nBuilding Compound & Multi-Key Index on 4.1M India targets...")
t0 = time.time()

idx_compact_name   = defaultdict(list)
idx_addr_num_token = defaultdict(list) # Compound: (number, addr_token)
idx_name_addr_num  = defaultdict(list) # Compound: (name_token, number)
idx_name_pair      = defaultdict(list) # Compound: (name_tok1, name_tok2)
idx_rare_name      = defaultdict(list) # Rare individual name token
idx_rare_addr      = defaultdict(list) # Rare individual address token

df_name_token = Counter()
df_addr_token = Counter()

# Count frequencies first
for row in tgt_india.itertuples(index=False):
    for t in get_name_tokens(row.business_name):
        df_name_token[t] += 1
    for t in get_addr_tokens(row.business_address):
        df_addr_token[t] += 1

# Populate indexes
for row in tgt_india.itertuples(index=False):
    eid = row.entity_id
    nm, ad = row.business_name, row.business_address
    
    # 1. Compact name
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4:
        idx_compact_name[cname].append(eid)
        
    ntoks = get_name_tokens(nm)
    anums = get_addr_numbers(ad)
    atoks = get_addr_tokens(ad)
    
    # 2. Compound: (num, addr_token) - ultra specific!
    for n in anums:
        for at in atoks[:4]:
            if df_addr_token[at] < 50000:
                idx_addr_num_token[(n, at)].append(eid)
                
    # 3. Compound: (name_tok, num) - ultra specific!
    for nt in ntoks[:3]:
        for n in anums[:3]:
            idx_name_addr_num[(nt, n)].append(eid)
            
    # 4. Compound: (name_tok1, name_tok2)
    if len(ntoks) >= 2:
        for i in range(min(3, len(ntoks))):
            for j in range(i+1, min(4, len(ntoks))):
                idx_name_pair[(ntoks[i], ntoks[j])].append(eid)
                
    # 5. Rare name tokens (DF < 2000)
    for nt in ntoks:
        if df_name_token[nt] < 2000:
            idx_rare_name[nt].append(eid)
            
    # 6. Rare address tokens (DF < 1500)
    for at in atoks:
        if df_addr_token[at] < 1500:
            idx_rare_addr[at].append(eid)

print(f"Indexes built in {time.time()-t0:.1f}s.")
print(f"  idx_compact_name: {len(idx_compact_name):,}")
print(f"  idx_addr_num_token: {len(idx_addr_num_token):,}")
print(f"  idx_name_addr_num: {len(idx_name_addr_num):,}")
print(f"  idx_name_pair: {len(idx_name_pair):,}")
print(f"  idx_rare_name: {len(idx_rare_name):,}")
print(f"  idx_rare_addr: {len(idx_rare_addr):,}")

# Query function
def query_compound(row, top_k=20):
    nm, ad = row.business_name, row.business_address
    scores = defaultdict(float)
    
    # 1. Exact compact name
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4 and cname in idx_compact_name:
        for cid in idx_compact_name[cname]:
            scores[cid] += 80.0
            
    ntoks = get_name_tokens(nm)
    anums = get_addr_numbers(ad)
    atoks = get_addr_tokens(ad)
    
    # 2. Compound: (num, addr_token)
    for n in anums:
        for at in atoks[:4]:
            key = (n, at)
            if key in idx_addr_num_token:
                cands = idx_addr_num_token[key]
                if len(cands) < 500:
                    weight = 40.0 / math.log(len(cands) + 2)
                    for cid in cands:
                        scores[cid] += weight
                        
    # 3. Compound: (name_tok, num)
    for nt in ntoks[:3]:
        for n in anums[:3]:
            key = (nt, n)
            if key in idx_name_addr_num:
                cands = idx_name_addr_num[key]
                if len(cands) < 500:
                    weight = 50.0 / math.log(len(cands) + 2)
                    for cid in cands:
                        scores[cid] += weight
                        
    # 4. Compound: (name_tok1, name_tok2)
    if len(ntoks) >= 2:
        for i in range(min(3, len(ntoks))):
            for j in range(i+1, min(4, len(ntoks))):
                key = (ntoks[i], ntoks[j])
                if key in idx_name_pair:
                    cands = idx_name_pair[key]
                    if len(cands) < 1000:
                        weight = 45.0 / math.log(len(cands) + 2)
                        for cid in cands:
                            scores[cid] += weight
                            
    # 5. Rare name token
    for nt in ntoks:
        if nt in idx_rare_name:
            cands = idx_rare_name[nt]
            weight = 30.0 / math.log(len(cands) + 2)
            for cid in cands:
                scores[cid] += weight
                
    # 6. Rare address token
    for at in atoks:
        if at in idx_rare_addr:
            cands = idx_rare_addr[at]
            weight = 25.0 / math.log(len(cands) + 2)
            for cid in cands:
                scores[cid] += weight
                
    if not scores:
        return []
        
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

# Test on 2000 queries
sample_s1 = s1_india[s1_india['entity_id'].map(lambda x: len(gt_dict.get(x, set())) > 0)].sample(2000, random_state=42)
print(f"\nQuerying {len(sample_s1)} entities with Compound Key Blocker...")

t0 = time.time()
recalls = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0, 30: 0}
total_true_matches = 0
cand_counts = []

for row in sample_s1.itertuples(index=False):
    true_matches = gt_dict.get(row.entity_id, set())
    total_true_matches += len(true_matches)
    
    ranked = query_compound(row, top_k=30)
    cand_counts.append(len(ranked))
    
    cand_ids = [cid for cid, s in ranked]
    for k in [5, 10, 15, 20, 25, 30]:
        top_k_ids = set(cand_ids[:k])
        recalls[k] += len(true_matches & top_k_ids)

q_time = time.time() - t0
print(f"2000 queries completed in {q_time:.1f}s ({len(sample_s1)/q_time:.1f} q/s)!")
print(f"Average candidates returned (cap=30): {np.mean(cand_counts):.1f}")
for k in [5, 10, 15, 20, 25, 30]:
    r = recalls[k] / total_true_matches
    print(f"  Recall @ Top-{k:2d}: {recalls[k]}/{total_true_matches} = {r*100:.2f}%")
