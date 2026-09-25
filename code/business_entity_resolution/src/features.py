"""
features.py
Feature engineering module computing 16-D pairwise similarity features
between reference entities and candidate target records using RapidFuzz.
"""

from typing import Dict, List, Set
from rapidfuzz import fuzz, distance
from normalizer import get_char_ngrams, tokenize, extract_digits


def compute_pair_features(s1_cache: Dict, t_meta: Dict, blocking_score: float) -> List[float]:
    """
    Computes a 16-dimensional dense feature vector for candidate pair (s1, target).
    s1_cache contains precomputed:
      'norm_name', 'norm_addr', 'digits', 'tokens', 'ngrams'
    t_meta contains precomputed:
      'norm_name', 'norm_addr', 'digits', 'missing_addr', 'is_s2'
    """
    s1_name = s1_cache['norm_name']
    t_name = t_meta['norm_name']

    # 1. Name string metrics
    n_rat = fuzz.ratio(s1_name, t_name) / 100.0
    n_prat = fuzz.partial_ratio(s1_name, t_name) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
    n_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0
    n_jw = distance.JaroWinkler.similarity(s1_name, t_name)

    # 2. Name subword & token overlaps
    t_ngrams = get_char_ngrams(t_name, n=3)
    u_ng = s1_cache['ngrams'] | t_ngrams
    n_ng_jacc = len(s1_cache['ngrams'] & t_ngrams) / len(u_ng) if u_ng else 0.0

    t_tokens = set(tokenize(t_name, min_len=2))
    u_tok = s1_cache['tokens'] | t_tokens
    n_tok_jacc = len(s1_cache['tokens'] & t_tokens) / len(u_tok) if u_tok else 0.0

    len_diff = float(abs(len(s1_name) - len(t_name)))

    # 3. Address string & structural metrics
    if t_meta['missing_addr']:
        a_rat = 0.0
        a_sort = 0.0
        a_set = 0.0
        dig_jacc = 0.0
        dig_exact = 0.0
    else:
        s1_addr = s1_cache['norm_addr']
        t_addr = t_meta['norm_addr']
        a_rat = fuzz.ratio(s1_addr, t_addr) / 100.0
        a_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        a_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

        u_dig = s1_cache['digits'] | t_meta['digits']
        dig_jacc = len(s1_cache['digits'] & t_meta['digits']) / len(u_dig) if u_dig else 0.0
        dig_exact = 1.0 if s1_cache['digits'] and s1_cache['digits'] == t_meta['digits'] else 0.0

    return [
        n_rat, n_prat, n_sort, n_set, n_jw, n_ng_jacc, n_tok_jacc, len_diff,
        a_rat, a_sort, a_set, dig_jacc, dig_exact,
        t_meta['missing_addr'], t_meta['is_s2'], float(blocking_score)
    ]
