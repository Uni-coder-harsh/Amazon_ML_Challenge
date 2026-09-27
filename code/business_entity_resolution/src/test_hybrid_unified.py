import pandas as pd
import numpy as np
import re
import unicodedata
import math
from collections import defaultdict, Counter
import time

TRAIN_DIR = "dataset/train"
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

print("Building unified multi-index...")
idx_compact_name   = defaultdict(list)
idx_name_token     = defaultdict(list)
idx_addr_num       = defaultdict(list)
idx_addr_token     = defaultdict(list)
idx_addr_num_token = defaultdict(list)

df_name_token = Counter()
df_addr_token = Counter()
df_addr_num   = Counter()

for row in tgt_india.itertuples(index=False):
    eid, nm, ad = row.entity_id, row.business_name, row.business_address
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4:
        idx_compact_name[cname].append(eid)
    ntoks = get_name_tokens(nm)
    for t in ntoks:
        idx_name_token[t].append(eid)
        df_name_token[t] += 1
    anums = get_addr_numbers(ad)
    for n in anums:
        idx_addr_num[n].append(eid)
        df_addr_num[n] += 1
    atoks = get_addr_tokens(ad)
    for t in atoks:
        idx_addr_token[t].append(eid)
        df_addr_token[t] += 1
    for n in anums:
        for at in atoks[:3]:
            if df_addr_token[at] < 50000:
                idx_addr_num_token[(n, at)].append(eid)

N_TARGETS = len(tgt_india)
def get_idf(df, N):
    return math.log((N + 1) / (df + 1)) + 1.0

def hybrid_query(row, top_k=25):
    nm, ad = row.business_name, row.business_address
    scores = defaultdict(float)
    
    # 1. Exact compact name
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4 and cname in idx_compact_name:
        for cid in idx_compact_name[cname]:
            scores[cid] += 100.0
            
    ntoks = get_name_tokens(nm)
    anums = get_addr_numbers(ad)
    atoks = get_addr_tokens(ad)
    
    # 2. Compound (num, addr_token)
    for n in anums:
        for at in atoks[:3]:
            key = (n, at)
            if key in idx_addr_num_token:
                cands = idx_addr_num_token[key]
                if len(cands) < 300:
                    w = 60.0 / math.log(len(cands) + 2)
                    for cid in cands:
                        scores[cid] += w
                        
    # 3. Name tokens
    for t in ntoks:
        df = df_name_token.get(t, 0)
        if 0 < df < 10000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_name_token[t]:
                scores[cid] += idf * 2.5
                
    # 4. Addr numbers
    for n in anums:
        df = df_addr_num.get(n, 0)
        if 0 < df < 15000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_addr_num[n]:
                scores[cid] += idf * 3.0
                
    # 5. Addr tokens
    for t in atoks:
        df = df_addr_token.get(t, 0)
        if 0 < df < 15000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_addr_token[t]:
                scores[cid] += idf * 2.5
                
    if not scores:
        return []
    return [cid for cid, s in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

sample_s1 = s1_india[s1_india['entity_id'].map(lambda x: len(gt_dict.get(x, set())) > 0)].sample(2000, random_state=42)
recalls = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0, 30: 0}
total_true = 0

for row in sample_s1.itertuples(index=False):
    tm = gt_dict.get(row.entity_id, set())
    total_true += len(tm)
    cands = hybrid_query(row, top_k=30)
    for k in [5, 10, 15, 20, 25, 30]:
        recalls[k] += len(tm & set(cands[:k]))

print(f"Results across {len(sample_s1)} entities ({total_true} true matches):")
for k in [5, 10, 15, 20, 25, 30]:
    r = recalls[k] / total_true
    print(f"  Top-{k:2d}: {recalls[k]}/{total_true} = {r*100:.2f}%")
