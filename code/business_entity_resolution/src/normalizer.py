"""
normalizer.py
─────────────
Robust text normalization that:
  - Handles Latin, Devanagari, Tamil, Telugu, French (UTF-8 preserved)
  - Canonicalizes legal suffixes (Pvt Ltd → '', Corp → '', etc.)
  - Normalizes address abbreviations (St → Street for token overlap)
  - Returns both a cleaned string AND a token-set for blocking
"""

import re
import unicodedata

# ── Legal suffix canonicalization ──────────────────────────────────────────
_LEGAL = re.compile(
    r"\b(pvt\.?|private|ltd\.?|limited|llc|l\.l\.c\.?|inc\.?|incorporated|"
    r"corp\.?|corporation|co\.?|llp|l\.l\.p\.?|lp|l\.p\.?|"
    r"gmbh|ag|sas|sarl|s\.a\.r\.l\.?|sa|s\.a\.?|pty|"
    r"plc|p\.l\.c\.?|bhd|sdn|nv|bv|cv|"
    r"enterprises?|enterprise|solutions?|solution|"
    r"services?|service|group|holding|holdings|"
    r"trading|traders?|industries|industry|"
    r"international|intl|india|pvt\.?\s*ltd\.?)\b",
    re.IGNORECASE,
)

# ── Address abbreviation expansion (for token overlap improvement) ─────────
_ADDR_ABBR = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave?\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bdist\b": "district",
    r"\bflr\b": "floor",
    r"\bfloor\b": "floor",
    r"\bno\b": "number",
    r"\bnagar\b": "nagar",
    r"\bph\b": "phase",
}

# ── Punctuation / symbol noise ─────────────────────────────────────────────
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")

# ── Stop tokens (too common to be useful for blocking) ────────────────────
_STOP = {
    "the", "and", "of", "in", "at", "to", "a", "an", "for", "by",
    "on", "with", "near", "shop", "store", "house", "centre", "center",
    "new", "old", "main", "big", "small", "no", "num", "number",
}


def normalize(text: str, expand_addr: bool = False) -> str:
    """
    Normalize text preserving Unicode (Indic scripts, accented French chars).
    Returns lowercase cleaned string.
    """
    if not text or not text.strip():
        return ""

    t = text.strip()

    # Normalize Unicode (NFC keeps Indic scripts intact; NFD would too)
    t = unicodedata.normalize("NFC", t)

    # Remove legal suffixes BEFORE lowercasing for pattern matching
    t = _LEGAL.sub(" ", t)

    # Lowercase
    t = t.lower()

    # Address abbreviation expansion
    if expand_addr:
        for pat, repl in _ADDR_ABBR.items():
            t = re.sub(pat, repl, t)

    # Remove punctuation (keep Unicode word chars and spaces)
    t = _PUNCT.sub(" ", t)

    # Collapse whitespace
    t = _SPACE.sub(" ", t).strip()

    return t


def get_tokens(text: str, min_len: int = 2, remove_stop: bool = True) -> set:
    """
    Returns a set of tokens from normalized text.
    Keeps Indic/French/Latin tokens as-is (Unicode-aware split).
    """
    n = normalize(text)
    tokens = set(n.split())
    tokens = {t for t in tokens if len(t) >= min_len}
    if remove_stop:
        tokens -= _STOP
    return tokens


def get_ngrams(text: str, n: int = 3) -> set:
    """
    Character n-grams from normalized text.
    Works for any script (Indic, Latin, etc.) since we work on Unicode chars.
    """
    norm = normalize(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def make_blocking_key(text: str) -> list:
    """
    Returns a list of blocking keys for inverted-index blocking.
    Keys are individual tokens (not stop words, len ≥ 3).
    """
    tokens = get_tokens(text, min_len=3, remove_stop=True)
    return list(tokens)


def extract_digits(text: str) -> list:
    """Extract digit sequences from address (house numbers, PIN codes)."""
    return re.findall(r"\d{2,}", text or "")
