"""
pairwise_features.py — Feature Engineering for Pairwise Classifier
===================================================================
28 features capturing name similarity, address similarity, embedding
cosine (when available), and structural signals.
"""

import re
import unicodedata
import numpy as np
from rapidfuzz import fuzz, distance as rfdist

# Reuse helpers from blocker
from blocker import deaccent, clean_name, name_tokens, addr_tokens, addr_numbers

N_FEATURES = 28

LEGAL_RE = re.compile(
    r"\b(pvt\.?|private|ltd\.?|limited|llc|inc\.?|corp\.?|co\.?|llp|lp|gmbh|ag|sas|sarl|sa)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    """Simple lowercased, de-accented, punctuation-stripped string."""
    s = deaccent(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compute_features(
    name_a: str, addr_a: str, country_a: str,
    name_b: str, addr_b: str, country_b: str,
    emb_a: np.ndarray | None = None,
    emb_b: np.ndarray | None = None,
) -> np.ndarray:
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    na = _norm(name_a)
    nb = _norm(name_b)
    aa = _norm(addr_a)
    ab = _norm(addr_b)

    empty_na = not na
    empty_nb = not nb
    empty_aa = not aa
    empty_ab = not ab

    # ── Name features (0-8) ───────────────────────────────────────────────
    if na and nb:
        feat[0]  = fuzz.token_sort_ratio(na, nb) / 100.0
        feat[1]  = fuzz.partial_ratio(na, nb)   / 100.0
        feat[2]  = fuzz.token_set_ratio(na, nb) / 100.0
        feat[3]  = rfdist.JaroWinkler.normalized_similarity(na, nb)
        feat[4]  = 1.0 - rfdist.Levenshtein.normalized_distance(na, nb)

        # Compact name match
        cna = clean_name(name_a)
        cnb = clean_name(name_b)
        if cna and cnb:
            feat[5] = 1.0 if cna == cnb else (
                1.0 if (len(cna) >= 4 and (cna in cnb or cnb in cna)) else 0.0
            )

        # Token Jaccard
        tna = set(name_tokens(name_a))
        tnb = set(name_tokens(name_b))
        feat[6] = len(tna & tnb) / max(len(tna | tnb), 1)

        # Length ratio
        feat[7] = min(len(na), len(nb)) / max(len(na), len(nb), 1)

    # Character 3-gram Jaccard of raw names
    def ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()
    ng_na = ngrams(na); ng_nb = ngrams(nb)
    feat[8] = len(ng_na & ng_nb) / max(len(ng_na | ng_nb), 1)

    # ── Address features (9-16) ───────────────────────────────────────────
    if aa and ab:
        feat[9]  = fuzz.token_sort_ratio(aa, ab)  / 100.0
        feat[10] = fuzz.partial_ratio(aa, ab)     / 100.0
        feat[11] = fuzz.token_set_ratio(aa, ab)   / 100.0
        feat[12] = min(len(aa), len(ab)) / max(len(aa), len(ab), 1)

    # Address number overlap
    dan = set(addr_numbers(addr_a))
    dbn = set(addr_numbers(addr_b))
    feat[13] = len(dan & dbn) / max(len(dan | dbn), 1)

    # Address token Jaccard
    taa = set(addr_tokens(addr_a))
    tab = set(addr_tokens(addr_b))
    feat[14] = len(taa & tab) / max(len(taa | tab), 1)

    # Both have same number
    feat[15] = 1.0 if (dan and dbn and dan & dbn) else 0.0

    # ── Cross-field features (16-20) ──────────────────────────────────────
    # Country match
    feat[16] = 1.0 if (country_a and country_b and country_a == country_b) else 0.0

    # Name token in address (e.g. company name mentioned in address)
    tna_set = set(name_tokens(name_a))
    for t in tna_set:
        if t in ab:
            feat[17] = 1.0
            break

    # Name compact substring of address
    cna2 = clean_name(name_a)
    if cna2 and len(cna2) >= 4 and cna2 in (aa + ab):
        feat[18] = 1.0

    # ── Embedding features (19-22) ────────────────────────────────────────
    if emb_a is not None and emb_b is not None:
        na_norm = np.linalg.norm(emb_a)
        nb_norm = np.linalg.norm(emb_b)
        if na_norm > 1e-8 and nb_norm > 1e-8:
            feat[19] = float(np.dot(emb_a, emb_b) / (na_norm * nb_norm))

    # ── Structural flags (20-27) ──────────────────────────────────────────
    feat[20] = 1.0 if empty_na else 0.0
    feat[21] = 1.0 if empty_nb else 0.0
    feat[22] = 1.0 if empty_aa else 0.0
    feat[23] = 1.0 if empty_ab else 0.0
    feat[24] = 1.0 if (empty_aa and empty_ab) else 0.0
    feat[25] = 1.0 if (not empty_aa and not empty_ab and feat[14] > 0.5) else 0.0  # strong addr
    feat[26] = 1.0 if (not empty_na and not empty_nb and feat[0] > 0.8) else 0.0  # strong name
    feat[27] = feat[25] * feat[26]  # both strong

    return feat


def compute_batch(pairs, lookupA: dict, lookupB: dict,
                  embs: dict | None = None) -> tuple[list, np.ndarray]:
    """
    pairs: list of (idA, idB)
    Returns (filtered_pairs, feature_matrix)
    """
    valid_pairs, rows = [], []
    for idA, idB in pairs:
        a = lookupA.get(idA)
        b = lookupB.get(idB)
        if a is None or b is None:
            continue
        ea = embs.get(idA) if embs else None
        eb = embs.get(idB) if embs else None
        feat = compute_features(
            a["name"], a["addr"], a["country"],
            b["name"], b["addr"], b["country"],
            emb_a=ea, emb_b=eb,
        )
        valid_pairs.append((idA, idB))
        rows.append(feat)
    X = np.array(rows, dtype=np.float32) if rows else np.zeros((0, N_FEATURES), np.float32)
    return valid_pairs, X
