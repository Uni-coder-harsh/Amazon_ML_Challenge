"""
normalizer.py
Robust, domain-agnostic text normalization and feature extraction engine
for multi-lingual business entity records (US, India, France).
"""

import re
import unicodedata
from typing import Set, List

# Common stop words across US, India, and France addresses/names
STOP_WORDS = {
    'the', 'and', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an',
    'de', 'la', 'le', 'et', 'du', 'des', 'en', 'les', 'par', 'pour', 'd', 'l',
    'near', 'opp', 'opposite', 'behind', 'beside', 'unit', 'suite', 'apt', 'floor',
    'road', 'street', 'ave', 'avenue', 'rd', 'st', 'lane', 'dr', 'drive', 'blvd'
}

LEGAL_SUFFIX_REPLACEMENTS = [
    (r'\b(private limited|pvt\.?\s*ltd\.?)\b', ' pvt ltd '),
    (r'\b(limited liability company|llc)\b', ' llc '),
    (r'\b(limited liability partnership|llp)\b', ' llp '),
    (r'\b(incorporated|inc\.?)\b', ' inc '),
    (r'\b(corporation|corp\.?)\b', ' corp '),
    (r'\b(limited|ltd\.?)\b', ' ltd '),
    (r'\b(company|co\.?)\b', ' co '),
    (r'\b(societe a responsabilite limitee|sarl)\b', ' sarl '),
    (r'\b(societe par actions simplifiee unipersonnelle|sasu)\b', ' sasu '),
    (r'\b(societe par actions simplifiee|sas)\b', ' sas '),
    (r'\b(societe anonyme|sa)\b', ' sa '),
    (r'\b(entreprise unipersonnelle a responsabilite limitee|eurl)\b', ' eurl '),
    (r'\b(societe civile immobiliere|sci)\b', ' sci '),
]

LEGAL_REGEXES = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in LEGAL_SUFFIX_REPLACEMENTS]
DOMAIN_RE = re.compile(r'https?://(?:www\.)?|\bwww\.', re.IGNORECASE)
EXT_RE = re.compile(r'\.(com|in|fr|org|net|co\.in|co|io|biz|info)\b', re.IGNORECASE)
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
DIGIT_RE = re.compile(r'\b\d+\b')


def normalize_text(text: str) -> str:
    """
    Decomposes Unicode (NFKD), folds accents, strips domain names,
    normalizes corporate suffixes, strips punctuation, and standardizes whitespace.
    """
    if not text:
        return ''
    # NFKD decomposition folds accents (é -> e, etc.)
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8').lower()
    text = DOMAIN_RE.sub('', text)
    text = EXT_RE.sub('', text)
    for rgx, repl in LEGAL_REGEXES:
        text = rgx.sub(repl, text)
    text = PUNCT_RE.sub(' ', text)
    return ' '.join(text.split())


def extract_digits(text: str) -> Set[str]:
    """
    Extracts all digit tokens from text (street numbers, PIN/zip codes).
    """
    if not text:
        return set()
    return set(DIGIT_RE.findall(str(text)))


def tokenize(text: str, min_len: int = 2) -> List[str]:
    """
    Tokenizes normalized string into meaningful non-stopword tokens.
    """
    if not text:
        return []
    tokens = text.split()
    return [t for t in tokens if len(t) >= min_len and t not in STOP_WORDS]


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Generates character n-grams with boundary markers.
    """
    if not text:
        return set()
    compact = ''.join(text.split())
    if len(compact) < n:
        return {compact}
    padded = f"^{compact}$"
    return {padded[i:i+n] for i in range(len(padded) - n + 1)}
