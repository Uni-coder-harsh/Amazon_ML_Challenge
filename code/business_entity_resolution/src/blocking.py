"""
blocking.py
───────────
Scored inverted-index blocking with ranking before candidate capping.

Key insight: naive union of token postings lists blows up for common tokens
like "technology", "pvt", "services" — leading to thousands of useless
candidates. The fix: count how many query tokens each candidate matches,
then rank by that score before capping to MAX_CANDIDATES.

Also separates name tokens (weight=2), address digits (weight=3, very
discriminative), and address location tokens (weight=1).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from normalizer import get_tokens, extract_digits, get_ngrams


class LexicalBlocker:
    """
    Scored inverted-index blocker.
    Candidate score = sum of matched token weights.
    Only top-MAX_CANDIDATES by score are returned.
    """

    def __init__(self, max_candidates: int = 40):
        self.max_candidates = max_candidates
        self._name_index:  dict[str, list[str]] = defaultdict(list)
        self._digit_index: dict[str, list[str]] = defaultdict(list)
        self._ngram_index: dict[str, list[str]] = defaultdict(list)

    def index(self, records: Iterable[tuple[str, str, str]]) -> "LexicalBlocker":
        """records: iterable of (entity_id, business_name, business_address)"""
        for eid, name, addr in records:
            for tok in get_tokens(name, min_len=3, remove_stop=True):
                self._name_index[tok].append(eid)
            for dig in extract_digits(addr):
                self._digit_index[dig].append(eid)
            for ng in get_ngrams(name, n=3):
                if len(ng) == 3:
                    self._ngram_index[ng].append(eid)
        return self

    def query(self, name: str, addr: str, use_ngrams: bool = True) -> set[str]:
        """
        Score every candidate by how many query features it matches,
        then return top max_candidates by score.
        """
        scores: dict[str, int] = {}

        def add(postings: list[str], weight: int):
            for eid in postings:
                scores[eid] = scores.get(eid, 0) + weight

        # Name tokens — weight 2 each
        name_tokens = get_tokens(name, min_len=3, remove_stop=True)
        for tok in name_tokens:
            add(self._name_index.get(tok, []), 2)

        # Address digit anchors — weight 3 each (highly discriminative)
        for dig in extract_digits(addr):
            add(self._digit_index.get(dig, []), 3)

        # N-gram fallback: only if name tokens gave very few results
        # (catches transliterations, domain names, abbreviations)
        if use_ngrams:
            n_strong = sum(1 for v in scores.values() if v >= 4)
            if n_strong < 10:
                for ng in get_ngrams(name, n=3):
                    if len(ng) == 3:
                        add(self._ngram_index.get(ng, []), 1)

        if not scores:
            return set()

        # Require score >= 2 (at least one name token match OR one digit match)
        # Fall back to score >= 1 if nothing qualifies
        min_score = 2
        ranked = sorted(
            [(eid, s) for eid, s in scores.items() if s >= min_score],
            key=lambda x: -x[1],
        )
        if not ranked:
            ranked = sorted(scores.items(), key=lambda x: -x[1])

        return {eid for eid, _ in ranked[: self.max_candidates]}

    def query_batch(
        self,
        queries: list[tuple[str, str, str]],
        use_ngrams: bool = True,
    ) -> dict[str, set[str]]:
        return {
            qid: self.query(name, addr, use_ngrams=use_ngrams)
            for qid, name, addr in queries
        }
