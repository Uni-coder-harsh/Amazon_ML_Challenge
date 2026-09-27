"""
features.py
───────────
Feature engineering for the pairwise matching classifier.
Uses both lexical similarity (rapidfuzz) and embedding cosine similarity.

Features (22 total):
  Lexical:
    0  name_token_sort_ratio      RapidFuzz token sort ratio (name)
    1  name_partial_ratio         RapidFuzz partial ratio (name)
    2  name_token_set_ratio       RapidFuzz token set ratio (name)
    3  name_jaro_winkler          Jaro-Winkler on normalized names
    4  addr_token_sort_ratio      Token sort ratio (address)
    5  addr_partial_ratio         Partial ratio (address)
    6  name_token_overlap         Jaccard on name token sets
    7  addr_token_overlap         Jaccard on address token sets
    8  name_ngram_overlap_3       Jaccard on 3-gram sets (name)
    9  addr_digit_overlap         Fraction of shared digit sequences
   10  country_match              1 if same country string
   11  name_len_ratio             min(len_a,len_b)/max(len_a,len_b)
   12  addr_len_ratio             same for address
   13  name_is_prefix             1 if shorter name is prefix of longer
   14  name_edit_dist_norm        1 - normalized Levenshtein distance

  Embedding:
   15  name_emb_cosine            Cosine sim of name embeddings (if available)
   16  addr_emb_cosine            Cosine sim of address embeddings (if available)
   17  combined_emb_cosine        Cosine sim of (name|addr) embeddings

  Structural:
   18  both_name_empty            1 if both names empty
   19  both_addr_empty            1 if both addresses empty
   20  name_a_empty               1 if query name empty
   21  name_b_empty               1 if target name empty
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, distance as rfdist
from normalizer import normalize, get_tokens, get_ngrams, extract_digits

N_FEATURES = 22


def compute_features(
    name_a: str,
    addr_a: str,
    name_b: str,
    addr_b: str,
    emb_a: np.ndarray | None = None,
    emb_b: np.ndarray | None = None,
    name_emb_a: np.ndarray | None = None,
    name_emb_b: np.ndarray | None = None,
    addr_emb_a: np.ndarray | None = None,
    addr_emb_b: np.ndarray | None = None,
    country_a: str = "",
    country_b: str = "",
) -> np.ndarray:
    """Compute feature vector for a pair (name_a, addr_a) vs (name_b, addr_b)."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    # Normalize
    na = normalize(name_a)
    nb = normalize(name_b)
    aa = normalize(addr_a, expand_addr=True)
    ab = normalize(addr_b, expand_addr=True)

    empty_na = len(na) == 0
    empty_nb = len(nb) == 0
    empty_aa = len(aa) == 0
    empty_ab = len(ab) == 0

    # ── Lexical name features ──────────────────────────────────────────────
    if not empty_na and not empty_nb:
        feat[0] = fuzz.token_sort_ratio(na, nb) / 100.0
        feat[1] = fuzz.partial_ratio(na, nb) / 100.0
        feat[2] = fuzz.token_set_ratio(na, nb) / 100.0
        feat[3] = rfdist.JaroWinkler.normalized_similarity(na, nb)
        # Edit distance
        max_len = max(len(na), len(nb))
        feat[14] = 1.0 - rfdist.Levenshtein.normalized_distance(na, nb)
        # Prefix check
        short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
        feat[13] = 1.0 if (short and long_.startswith(short)) else 0.0
        # Len ratio
        if max_len > 0:
            feat[11] = min(len(na), len(nb)) / max_len
    else:
        # partial info: at least set len ratio
        max_len = max(len(na), len(nb)) if (na or nb) else 1
        if max_len > 0:
            feat[11] = min(len(na), len(nb)) / max_len

    # ── Lexical address features ───────────────────────────────────────────
    if not empty_aa and not empty_ab:
        feat[4] = fuzz.token_sort_ratio(aa, ab) / 100.0
        feat[5] = fuzz.partial_ratio(aa, ab) / 100.0
        max_len_a = max(len(aa), len(ab))
        if max_len_a > 0:
            feat[12] = min(len(aa), len(ab)) / max_len_a

    # ── Token overlap features ─────────────────────────────────────────────
    tna = get_tokens(name_a, min_len=2, remove_stop=False)
    tnb = get_tokens(name_b, min_len=2, remove_stop=False)
    if tna or tnb:
        feat[6] = len(tna & tnb) / max(len(tna | tnb), 1)

    taa = get_tokens(addr_a, min_len=2, remove_stop=False)
    tab = get_tokens(addr_b, min_len=2, remove_stop=False)
    if taa or tab:
        feat[7] = len(taa & tab) / max(len(taa | tab), 1)

    # ── N-gram overlap (name) ─────────────────────────────────────────────
    ng_na = get_ngrams(name_a, n=3)
    ng_nb = get_ngrams(name_b, n=3)
    if ng_na or ng_nb:
        feat[8] = len(ng_na & ng_nb) / max(len(ng_na | ng_nb), 1)

    # ── Digit overlap (address) ───────────────────────────────────────────
    da = set(extract_digits(addr_a))
    db = set(extract_digits(addr_b))
    if da or db:
        feat[9] = len(da & db) / max(len(da | db), 1)

    # ── Country match ─────────────────────────────────────────────────────
    feat[10] = 1.0 if (country_a and country_b and country_a == country_b) else 0.0

    # ── Embedding cosine similarities ─────────────────────────────────────
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na_ = np.linalg.norm(a)
        nb_ = np.linalg.norm(b)
        if na_ < 1e-8 or nb_ < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na_ * nb_))

    feat[15] = cosine(name_emb_a, name_emb_b)
    feat[16] = cosine(addr_emb_a, addr_emb_b)
    feat[17] = cosine(emb_a, emb_b)

    # ── Structural flags ──────────────────────────────────────────────────
    feat[18] = 1.0 if (empty_na and empty_nb) else 0.0
    feat[19] = 1.0 if (empty_aa and empty_ab) else 0.0
    feat[20] = 1.0 if empty_na else 0.0
    feat[21] = 1.0 if empty_nb else 0.0

    return feat


def compute_features_batch(
    pairs: list[tuple],
    embeddings: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """
    pairs: list of (id_a, name_a, addr_a, country_a, id_b, name_b, addr_b, country_b)
    embeddings: optional dict entity_id → embedding vector (combined name|addr)
    Returns: (N, N_FEATURES) feature matrix
    """
    n = len(pairs)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i, (id_a, na, aa, ca, id_b, nb, ab, cb) in enumerate(pairs):
        emb_a = embeddings.get(id_a) if embeddings else None
        emb_b = embeddings.get(id_b) if embeddings else None
        X[i] = compute_features(na, aa, nb, ab, emb_a=emb_a, emb_b=emb_b,
                                 country_a=ca, country_b=cb)
    return X
