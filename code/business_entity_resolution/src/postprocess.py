"""
postprocess.py
──────────────
Post-processing for entity resolution output:
  1. No cross-country matches (verified empirically: 0 in training)
  2. Injectivity: a S2/S3 ID can only appear in ONE S1's match list
     (enforced by keeping the highest-probability assignment)
  3. Singleton detection: entities with no matches get empty matched_entity_ids
"""

from __future__ import annotations

from collections import defaultdict


def enforce_injectivity(
    matches: dict[str, set[str]],
    probs: dict[tuple[str, str], float],
) -> dict[str, set[str]]:
    """
    Ensure each S2/S3 ID maps to at most one S1.
    When there's a conflict, keep the (s1, s23) pair with the highest probability.
    
    Args:
        matches: s1_id → set of matched s23_ids
        probs:   (s1_id, s23_id) → match probability
    
    Returns:
        Filtered matches dict with injectivity enforced.
    """
    # Build reverse: s23_id → list of (s1_id, prob)
    reverse: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s1_id, s23_ids in matches.items():
        for s23_id in s23_ids:
            prob = probs.get((s1_id, s23_id), 0.5)
            reverse[s23_id].append((s1_id, prob))
    
    # Find conflicts and resolve
    bad_pairs: set[tuple[str, str]] = set()  # (s1_id, s23_id) to remove
    for s23_id, assignments in reverse.items():
        if len(assignments) > 1:
            # Keep the highest-prob assignment
            best_s1 = max(assignments, key=lambda x: x[1])[0]
            for s1_id, _ in assignments:
                if s1_id != best_s1:
                    bad_pairs.add((s1_id, s23_id))
    
    # Apply removals
    result: dict[str, set[str]] = {}
    for s1_id, s23_ids in matches.items():
        result[s1_id] = {sid for sid in s23_ids if (s1_id, sid) not in bad_pairs}
    
    return result


def filter_cross_country(
    matches: dict[str, set[str]],
    s1_country: dict[str, str],
    s23_country: dict[str, str],
) -> dict[str, set[str]]:
    """Remove any cross-country matches (empirically impossible in training data)."""
    result: dict[str, set[str]] = {}
    for s1_id, s23_ids in matches.items():
        country = s1_country.get(s1_id, "")
        result[s1_id] = {
            sid for sid in s23_ids
            if not country or s23_country.get(sid, "") == country
        }
    return result
