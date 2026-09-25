"""
blocking.py
High-throughput multi-key inverted index blocking engine.
Generates candidate pairs per country with high recall (Pairs Completeness >= 0.98).
"""

from collections import defaultdict
from typing import List, Dict, Set, Tuple
from normalizer import normalize_text, tokenize, extract_digits


class MultiKeyBlocker:
    """
    Multi-Key Inverted Index Blocker.
    Indexes target records (S2 + S3) by:
      1. Normalized Name Core Tokens (length >= 3)
      2. Address Digits / Numbers (house numbers, PIN codes)
      3. Address Distinctive Locality Tokens (length >= 4)
    """

    def __init__(self, max_postings_token: int = 1500, max_postings_digit: int = 800, top_k: int = 25):
        self.max_postings_token = max_postings_token
        self.max_postings_digit = max_postings_digit
        self.top_k = top_k
        self.name_token_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        self.target_records: List[Dict] = []
        self.target_metadata: List[Dict] = []

    def fit(self, targets: List[Dict]):
        """
        Builds inverted indexes over target records.
        targets is a list of dicts with keys: 'entity_id', 'business_name', 'business_address'
        """
        self.target_records = targets
        self.target_metadata = []
        self.name_token_index.clear()
        self.addr_token_index.clear()
        self.digit_index.clear()

        for idx, r in enumerate(targets):
            nn = normalize_text(r.get('business_name', ''))
            na = normalize_text(r.get('business_address', ''))
            digits = extract_digits(na)
            tokens = tokenize(nn, min_len=3)
            addr_tokens = tokenize(na, min_len=4)

            self.target_metadata.append({
                'entity_id': r['entity_id'],
                'norm_name': nn,
                'norm_addr': na,
                'digits': digits,
                'missing_addr': 1.0 if not na else 0.0,
                'is_s2': 1.0 if r['entity_id'].startswith('S2-') else 0.0
            })

            for t in tokens:
                self.name_token_index[t].append(idx)
            for d in digits:
                self.digit_index[d].append(idx)
            for at in addr_tokens:
                self.addr_token_index[at].append(idx)

    def retrieve_candidates(self, s1_name: str, s1_addr: str) -> List[Tuple[int, int]]:
        """
        Retrieves top candidate indices and initial scores for an S1 entity.
        Returns list of (target_idx, lexical_score).
        """
        s1_nn = normalize_text(s1_name)
        s1_na = normalize_text(s1_addr)
        s1_digits = extract_digits(s1_na)
        s1_tokens = tokenize(s1_nn, min_len=3)
        s1_addr_tokens = tokenize(s1_na, min_len=4)

        cand_scores = defaultdict(int)

        # Name token match (weight 3)
        for t in s1_tokens:
            postings = self.name_token_index.get(t)
            if postings and len(postings) <= self.max_postings_token:
                for p in postings:
                    cand_scores[p] += 3

        # Addr digits match (weight 3)
        for d in s1_digits:
            postings = self.digit_index.get(d)
            if postings and len(postings) <= self.max_postings_digit:
                for p in postings:
                    cand_scores[p] += 3

        # Addr tokens match (weight 2)
        for at in s1_addr_tokens:
            postings = self.addr_token_index.get(at)
            if postings and len(postings) <= self.max_postings_digit:
                for p in postings:
                    cand_scores[p] += 2

        if not cand_scores:
            return []

        # Return top_k candidates sorted by lexical score
        top_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return top_cands
