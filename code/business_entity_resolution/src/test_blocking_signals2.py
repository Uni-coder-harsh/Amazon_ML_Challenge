import pandas as pd
import numpy as np
import re
from collections import Counter

TRAIN_DIR = "dataset/train"
tr1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str).fillna("")
tr2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str).fillna("")
tr3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str).fillna("")
gt  = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
)

s1_lkp = tr1.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
s23_df = pd.concat([tr2, tr3], ignore_index=True)
s23_lkp = s23_df.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')

# Sample 20,000 ground truth S1 with matches
sample_gt = gt[gt['match_list'].apply(len) > 0].sample(20000, random_state=42)

def clean_alnum(s):
    s = s.lower()
    s = re.sub(r"\b(pvt|private|ltd|limited|llc|inc|corp|corporation|co|llp|lp|gmbh|ag|sas|sarl|sa)\b", "", s)
    s = re.sub(r"(\.com|\.in|\.org|\.net|\.io|@|#)", "", s)
    return re.sub(r"[^a-z0-9]", "", s)

def extract_numbers(s):
    return set(re.findall(r"\d+", s))

def clean_addr_words(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    words = set(w for w in s.split() if len(w) >= 3)
    stopwords = {'the', 'and', 'near', 'opp', 'opposite', 'road', 'street', 'floor', 'flat', 'house', 'plot', 'shop', 'building'}
    return words - stopwords

def clean_name_words(s):
    s = s.lower()
    s = re.sub(r"\b(pvt|private|ltd|limited|llc|inc|corp|corporation|co|llp|lp|gmbh|ag|sas|sarl|sa)\b", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    words = set(w for w in s.split() if len(w) >= 4)
    stopwords = {'enterprises', 'enterprise', 'services', 'service', 'solutions', 'solution', 'group', 'company', 'india', 'international'}
    return words - stopwords

total_matches = 0
covered = 0
uncovered = 0
uncovered_examples = []

for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1 = s1_lkp[s1_id]
    s1_cname = clean_alnum(s1['business_name'])
    s1_nums = extract_numbers(s1['business_address'])
    s1_awords = clean_addr_words(s1['business_address'])
    s1_nwords = clean_name_words(s1['business_name'])
    
    for m_id in row['match_list']:
        if m_id not in s23_lkp:
            continue
        m = s23_lkp[m_id]
        total_matches += 1
        
        m_cname = clean_alnum(m['business_name'])
        m_nums = extract_numbers(m['business_address'])
        m_awords = clean_addr_words(m['business_address'])
        m_nwords = clean_name_words(m['business_name'])
        
        is_name_match = (s1_cname and m_cname and (s1_cname == m_cname))
        is_name_sub = (s1_cname and m_cname and (s1_cname in m_cname or m_cname in s1_cname) and min(len(s1_cname), len(m_cname)) >= 4)
        is_num_overlap = len(s1_nums & m_nums) > 0
        is_aword_overlap = len(s1_awords & m_awords) >= 2
        is_nword_overlap = len(s1_nwords & m_nwords) >= 1
        
        if is_name_match or is_name_sub or is_num_overlap or is_aword_overlap or is_nword_overlap:
            covered += 1
        else:
            uncovered += 1
            if len(uncovered_examples) < 10:
                uncovered_examples.append((s1['business_name'], s1['business_address'], m['business_name'], m['business_address']))

print(f"Total true match pairs analyzed: {total_matches}")
print(f"COVERED: {covered} ({covered/total_matches*100:.3f}%)")
print(f"UNCOVERED: {uncovered} ({uncovered/total_matches*100:.3f}%)")
print(f"\nRemaining uncovered count: {uncovered}")
for s1n, s1a, mn, ma in uncovered_examples:
    print(f"  S1: {s1n} | {s1a}")
    print(f"  M : {mn} | {ma}")
    print()
