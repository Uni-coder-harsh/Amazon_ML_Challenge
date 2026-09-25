"""
postprocess.py
Mathematical optimization and post-processing engine:
- F_0.5 decision thresholding
- Singleton gate protection
- Single-parent (injectivity) greedy max-margin conflict resolution
- Submission formatting and subset verification
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict


def resolve_matches_and_conflicts(
    predictions_by_s1: Dict[str, List[Tuple[str, float]]],
    threshold: float = 0.70,
    singleton_gate: float = 0.35
) -> Dict[str, List[str]]:
    """
    Applies decision threshold, singleton gate, and injectivity constraint.
    predictions_by_s1: Dict[s1_id -> list of (target_id, score)]
    Returns: Dict[s1_id -> list of matched target_ids]
    """
    # 1. Filter by threshold & singleton gate
    tentative_assignments = defaultdict(list)
    target_to_s1_scores = defaultdict(list) # target_id -> list of (score, s1_id)

    for s1_id, cand_pairs in predictions_by_s1.items():
        if not cand_pairs:
            continue
        max_score = max(score for _, score in cand_pairs)
        if max_score < singleton_gate:
            continue # singleton gate: leave empty
        for tid, score in cand_pairs:
            if score >= threshold:
                tentative_assignments[s1_id].append((tid, score))
                target_to_s1_scores[tid].append((score, s1_id))

    # 2. Enforce Single-Parent Invariant (Injectivity)
    # A target can belong to at most ONE S1 entity.
    # If conflict arises, assign to argmax_s1 P(s1, target).
    winner_for_target = {}
    for tid, s1_list in target_to_s1_scores.items():
        if len(s1_list) == 1:
            winner_for_target[tid] = s1_list[0][1]
        else:
            # Sort descending by score
            s1_list.sort(key=lambda x: x[0], reverse=True)
            winner_for_target[tid] = s1_list[0][1]

    # 3. Build final matched lists
    final_matches = defaultdict(list)
    for s1_id, pairs in tentative_assignments.items():
        for tid, score in pairs:
            if winner_for_target.get(tid) == s1_id:
                final_matches[s1_id].append(tid)

    return final_matches
