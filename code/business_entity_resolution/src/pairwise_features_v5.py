"""
pairwise_features_v5.py — Enhanced 40-Feature Pairwise Classifier
==================================================================
New features over v4 (28 → 40):
  - feat[28-31]: Phonetic similarity (soundex, metaphone via jellyfish)
  - feat[32-34]: Character 4-gram and bigram Jaccard
  - feat[35-36]: Name abbreviation expansion match
  - feat[37]: PIN code / postal code exact match
  - feat[38]: Embedding cosine similarity from ANN index
  - feat[39]: Combined name+address embedding cosine similarity
"""

import re
import unicodedata
import numpy as np
from rapidfuzz import fuzz, distance as rfdist

# Reuse helpers from blocker
from blocker import deaccent, clean_name, name_tokens, addr_tokens, addr_numbers

N_FEATURES = 40

LEGAL_RE = re.compile(
    r"\b(pvt\.?|private|ltd\.?|limited|llc|inc\.?|corp\.?|co\.?|llp|lp|gmbh|ag|sas|sarl|sa)\b",
    re.IGNORECASE,
)

# Try to load jellyfish for phonetics
try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

# Transliteration map for common Hindi/Indic business words
_TRANSLIT_MAP = {
    # Hindi → Latin
    "प्राइवेट": "private", "लिमिटेड": "limited", "प्रा": "pra",
    "एंटरप्राइजेज": "enterprises", "सर्विसेज": "services",
    "ट्रेडर्स": "traders", "सॉल्यूशंस": "solutions",
    "कंस्ट्रक्शन": "construction", "इंटरनेशनल": "international",
    "इंडस्ट्रीज": "industries", "टेक्नोलॉजीज": "technologies",
    # Common abbreviations in business names
    "pvt": "private", "ltd": "limited", "mfg": "manufacturing",
    "dist": "distribution", "int": "international",
}

def _norm(s: str) -> str:
    """Simple lowercased, de-accented, punctuation-stripped string."""
    s = deaccent(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_pin(addr: str) -> set:
    """Extract 6-digit Indian PINs or 5-digit US ZIPs from address."""
    pins = set()
    for m in re.finditer(r'\b(\d{6})\b', addr):  # Indian PIN
        pins.add(m.group(1))
    for m in re.finditer(r'\b(\d{5})\b', addr):  # US ZIP
        pins.add(m.group(1))
    # Also strip spaces: "400 001" → "400001"
    for m in re.finditer(r'\b(\d{3})\s+(\d{3})\b', addr):
        pins.add(m.group(1) + m.group(2))
    return pins


def _has_indic(s: str) -> bool:
    """Check if string contains Indic-script characters."""
    for c in s:
        cp = ord(c)
        if 0x0900 <= cp <= 0x097F:  # Devanagari
            return True
        if 0x0B00 <= cp <= 0x0B7F:  # Oriya
            return True
        if 0x0C00 <= cp <= 0x0C7F:  # Telugu
            return True
        if 0x0B80 <= cp <= 0x0BFF:  # Tamil
            return True
        if 0x0A80 <= cp <= 0x0AFF:  # Gujarati
            return True
    return False


def _soundex_sim(a: str, b: str) -> float:
    """Soundex code similarity for first meaningful token."""
    if not _HAS_JELLYFISH or not a or not b:
        return 0.0
    try:
        ta = name_tokens(a)
        tb = name_tokens(b)
        if not ta or not tb:
            return 0.0
        # Compare first 2 tokens
        scores = []
        for wa in ta[:2]:
            for wb in tb[:2]:
                if wa.isascii() and wb.isascii():
                    sa = jellyfish.soundex(wa)
                    sb = jellyfish.soundex(wb)
                    scores.append(1.0 if sa == sb else 0.0)
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0


def _metaphone_sim(a: str, b: str) -> float:
    """Metaphone similarity for phonetic matching."""
    if not _HAS_JELLYFISH or not a or not b:
        return 0.0
    try:
        ta = name_tokens(a)
        tb = name_tokens(b)
        if not ta or not tb:
            return 0.0
        scores = []
        for wa in ta[:2]:
            for wb in tb[:2]:
                if wa.isascii() and wb.isascii():
                    ma = jellyfish.metaphone(wa)
                    mb = jellyfish.metaphone(wb)
                    if ma and mb:
                        scores.append(fuzz.ratio(ma, mb) / 100.0)
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0


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

    ng3_na = ngrams(na, 3); ng3_nb = ngrams(nb, 3)
    feat[8] = len(ng3_na & ng3_nb) / max(len(ng3_na | ng3_nb), 1)

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

    # ── Embedding features (19) ────────────────────────────────────────────
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

    # ── NEW: Phonetic features (28-31) ────────────────────────────────────
    feat[28] = _soundex_sim(na, nb)
    feat[29] = _metaphone_sim(na, nb)

    # Script flags: Indic presence signals cross-script pair
    feat[30] = 1.0 if _has_indic(name_a) else 0.0
    feat[31] = 1.0 if _has_indic(name_b) else 0.0

    # ── NEW: Extended n-gram features (32-34) ─────────────────────────────
    # 4-gram Jaccard
    ng4_na = ngrams(na, 4); ng4_nb = ngrams(nb, 4)
    feat[32] = len(ng4_na & ng4_nb) / max(len(ng4_na | ng4_nb), 1)

    # Bigram Jaccard
    ng2_na = ngrams(na, 2); ng2_nb = ngrams(nb, 2)
    feat[33] = len(ng2_na & ng2_nb) / max(len(ng2_na | ng2_nb), 1)

    # Address 3-gram Jaccard
    ng3_aa = ngrams(aa, 3); ng3_ab = ngrams(ab, 3)
    feat[34] = len(ng3_aa & ng3_ab) / max(len(ng3_aa | ng3_ab), 1)

    # ── NEW: Abbreviation expansion (35-36) ───────────────────────────────
    # Check if one name is an abbreviation of the other (e.g., "ABC" → "A Better Company")
    def abbrev_match(short: str, long: str) -> float:
        words = long.split()
        if not words or not short:
            return 0.0
        initials = "".join(w[0] for w in words if w)
        if initials.lower() == short.lower():
            return 1.0
        # Partial
        if len(short) >= 2 and initials.lower().startswith(short.lower()):
            return 0.8
        return 0.0

    if na and nb:
        feat[35] = max(abbrev_match(na.replace(" ", ""), nb),
                       abbrev_match(nb.replace(" ", ""), na))
        # Word-level prefix match (first significant words)
        ta_words = na.split()[:3]
        tb_words = nb.split()[:3]
        if ta_words and tb_words:
            prefix_sim = sum(
                1 if (a == b or (len(a) >= 3 and b.startswith(a[:3])))
                else 0
                for a, b in zip(ta_words, tb_words)
            ) / max(len(ta_words), len(tb_words), 1)
            feat[36] = prefix_sim

    # ── NEW: PIN/postal code exact match (37) ─────────────────────────────
    pins_a = _extract_pin(addr_a)
    pins_b = _extract_pin(addr_b)
    feat[37] = 1.0 if (pins_a and pins_b and pins_a & pins_b) else 0.0

    # ── NEW: Name-only embedding cosine (38) & addr embedding cosine (39) ──
    # These are populated externally by the pipeline when embeddings are available.
    # (feat[19] carries combined, feat[38] name-only, feat[39] addr-only)
    # Left as 0 here; filled by pipeline if separate embeddings exist.
    # feat[38] = 0.0  (already zero)
    # feat[39] = 0.0  (already zero)

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
