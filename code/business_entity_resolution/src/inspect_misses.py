import pandas as pd
import numpy as np
import re
import unicodedata
import math
from collections import defaultdict, Counter

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
s23_lkp = tgt_india.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')

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
    nums = set()
    for n in re.findall(r"\d+", s):
        clean_n = n.lstrip('0')
        if clean_n and len(clean_n) >= 2:
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

idx_compact_name = defaultdict(list)
idx_name_token   = defaultdict(list)
idx_addr_num     = defaultdict(list)
idx_addr_token   = defaultdict(list)

df_name_token = Counter()
df_addr_token = Counter()
df_addr_num   = Counter()

for row in tgt_india.itertuples(index=False):
    eid, nm, ad = row.entity_id, row.business_name, row.business_address
    cname = get_clean_compact_name(nm)
    if cname and len(cname) >= 4:
        idx_compact_name[cname].append(eid)
    for t in get_name_tokens(nm):
        idx_name_token[t].append(eid)
        df_name_token[t] += 1
    for n in get_addr_numbers(ad):
        idx_addr_num[n].append(eid)
        df_addr_num[n] += 1
    for t in get_addr_tokens(ad):
        idx_addr_token[t].append(eid)
        df_addr_token[t] += 1

N_TARGETS = len(tgt_india)
def get_idf(df, N):
    return math.log((N + 1) / (df + 1)) + 1.0

# Analyze 500 S1 queries
sample_s1 = s1_india[s1_india['entity_id'].map(lambda x: len(gt_dict.get(x, set())) > 0)].sample(500, random_state=42)

misses = []
for row in sample_s1.itertuples(index=False):
    true_matches = {m for m in gt_dict.get(row.entity_id, set()) if m in s23_lkp}
    if not true_matches:
        continue
        
    candidate_scores = defaultdict(float)
    cname = get_clean_compact_name(row.business_name)
    if cname and len(cname) >= 4 and cname in idx_compact_name:
        for cid in idx_compact_name[cname]:
            candidate_scores[cid] += 50.0
            
    for t in get_name_tokens(row.business_name):
        df = df_name_token.get(t, 0)
        if 0 < df < 10000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_name_token[t]:
                candidate_scores[cid] += idf * 2.0
                
    for n in get_addr_numbers(row.business_address):
        df = df_addr_num.get(n, 0)
        if 0 < df < 15000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_addr_num[n]:
                candidate_scores[cid] += idf * 2.5
                
    for t in get_addr_tokens(row.business_address):
        df = df_addr_token.get(t, 0)
        if 0 < df < 20000:
            idf = get_idf(df, N_TARGETS)
            for cid in idx_addr_token[t]:
                candidate_scores[cid] += idf * 1.0
                
    ranked = [cid for cid, s in sorted(candidate_scores.items(), key=lambda x: -x[1])[:30]]
    for tm in true_matches:
        if tm not in ranked:
            m_info = s23_lkp[tm]
            score_in_pool = candidate_scores.get(tm, 0.0)
            misses.append((row.business_name, row.business_address, m_info['business_name'], m_info['business_address'], score_in_pool, len(candidate_scores)))

print(f"Total misses analyzed: {len(misses)}")
for s1n, s1a, mn, ma, sc, pool_sz in misses[:15]:
    print(f"S1: {s1n} | {s1a}")
    print(f"TM: {mn} | {ma}")
    print(f"    Score in pool: {sc:.1f} | Total cands generated: {pool_sz}")
    print("-" * 60)
