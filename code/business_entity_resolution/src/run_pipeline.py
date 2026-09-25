"""
run_pipeline.py
High-Performance Master Execution Pipeline for Business Entity Resolution:
1. Fast Training & Threshold Optimization on Training Data (or load pre-trained artifact)
2. Multi-Key Inverted Index Blocking per Country Partition with Parquet Checkpointing
3. RapidFuzz Pairwise Feature Extraction & Vectorized LightGBM Inference
4. Injectivity & Singleton Gating Post-Processing
5. Dual Output Generation (matching_results.tsv & candidate_pairs.tsv)
6. Automated Validation with validate_submission.py
"""

import os
import sys
import time
import argparse
import numpy as np
import polars as pl
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from normalizer import normalize_text, tokenize, extract_digits, get_char_ngrams
from blocking import MultiKeyBlocker
from features import compute_pair_features
from model import build_model, save_model, load_model
from postprocess import resolve_matches_and_conflicts


def train_model_pipeline(
    train_dir: str,
    artifacts_dir: str,
    sample_entities: int = 30000,
    top_k: int = 25
) -> Tuple[object, float]:
    """
    Trains LightGBM matcher on a balanced sample of training data.
    Finds optimal decision threshold tau* maximizing Macro-F_0.5.
    """
    print(f"\n[1/6] Training Phase: Sampling {sample_entities} entities from {train_dir}...")
    t0 = time.time()

    # Load ground truth sample
    gt_df = pl.scan_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), separator='\t')\
        .head(sample_entities).collect()

    s1_ids = gt_df['source1_entity_id'].to_list()
    gt_map = {
        r['source1_entity_id']: set(r['matched_entity_ids'].split(',')) if r['matched_entity_ids'] else set()
        for r in gt_df.to_dicts()
    }

    # Load S1 records
    s1_df = pl.scan_csv(os.path.join(train_dir, 'train_source1.tsv'), separator='\t')\
        .filter(pl.col('entity_id').is_in(s1_ids)).collect()

    all_target_ids = []
    for mids in gt_map.values():
        all_target_ids.extend(list(mids))

    print(f"Loaded {len(s1_df)} reference entities and {len(all_target_ids)} true target links.")

    # Load target records (true targets + 40,000 background distractors)
    s2_df = pl.scan_csv(os.path.join(train_dir, 'train_source2.tsv'), separator='\t')\
        .filter(pl.col('entity_id').is_in(all_target_ids)).collect()
    s3_df = pl.scan_csv(os.path.join(train_dir, 'train_source3.tsv'), separator='\t')\
        .filter(pl.col('entity_id').is_in(all_target_ids)).collect()

    s2_dist = pl.scan_csv(os.path.join(train_dir, 'train_source2.tsv'), separator='\t').head(25000).collect()
    s3_dist = pl.scan_csv(os.path.join(train_dir, 'train_source3.tsv'), separator='\t').head(25000).collect()

    target_pool = pl.concat([s2_df, s3_df, s2_dist, s3_dist]).unique(subset=['entity_id']).to_dicts()
    print(f"Target pool constructed with {len(target_pool)} records in {time.time() - t0:.2f}s.")

    # Fit Blocker
    blocker = MultiKeyBlocker(max_postings_token=1000, max_postings_digit=500, top_k=top_k)
    blocker.fit(target_pool)

    # Generate candidate pairs & features
    print("Extracting pairwise features for training...")
    t_feat = time.time()
    X_list, y_list, pair_meta = [], [], []

    for s1_r in s1_df.to_dicts():
        s1_id = s1_r['entity_id']
        true_mids = gt_map.get(s1_id, set())

        s1_cache = {
            'norm_name': normalize_text(s1_r['business_name']),
            'norm_addr': normalize_text(s1_r['business_address']),
            'digits': extract_digits(normalize_text(s1_r['business_address'])),
            'tokens': set(tokenize(normalize_text(s1_r['business_name']), min_len=2)),
            'ngrams': get_char_ngrams(normalize_text(s1_r['business_name']), n=3)
        }

        cands = blocker.retrieve_candidates(s1_r['business_name'], s1_r['business_address'])
        for tidx, b_score in cands:
            t_meta = blocker.target_metadata[tidx]
            feat = compute_pair_features(s1_cache, t_meta, b_score)
            X_list.append(feat)
            y_list.append(1 if t_meta['entity_id'] in true_mids else 0)
            pair_meta.append((s1_id, t_meta['entity_id']))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Feature matrix generated: {X.shape} (Positives: {np.sum(y)}, Negatives: {len(y) - np.sum(y)}) in {time.time() - t_feat:.2f}s.")

    # Validation Split (by S1 cluster)
    unique_s1 = list(gt_map.keys())
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    split_idx = int(len(unique_s1) * 0.75)
    train_s1_set = set(unique_s1[:split_idx])
    val_s1_set = set(unique_s1[split_idx:])

    train_mask = np.array([s1 in train_s1_set for s1, _ in pair_meta])
    val_mask = ~train_mask

    # Train LightGBM model
    print("Fitting LightGBM Classifier...")
    clf = build_model(n_estimators=180, learning_rate=0.08, num_leaves=31, random_state=42)
    clf.fit(X[train_mask], y[train_mask])

    # Optimize Threshold on Validation Set
    val_probs = clf.predict_proba(X[val_mask])[:, 1]
    val_indices = np.where(val_mask)[0]

    val_cand_map = defaultdict(list)
    for i_val, i_orig in enumerate(val_indices):
        s1_id, tid = pair_meta[i_orig]
        val_cand_map[s1_id].append((tid, val_probs[i_val]))

    best_tau, best_f05 = 0.60, 0.0
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        resolved = resolve_matches_and_conflicts(val_cand_map, threshold=tau, singleton_gate=0.35)
        scores = []
        for s1_id in val_s1_set:
            y_t = gt_map.get(s1_id, set())
            y_p = set(resolved.get(s1_id, []))
            if not y_t:
                scores.append(1.0 if not y_p else 0.0)
            else:
                if not y_p:
                    scores.append(0.0)
                else:
                    tp = len(y_t & y_p)
                    scores.append((5.0 * tp) / (len(y_t) + 4.0 * len(y_p)))
        mean_score = float(np.mean(scores))
        print(f"  Validation tau={tau:.2f} -> Macro F_0.5 = {mean_score:.4f}")
        if mean_score > best_f05:
            best_f05 = mean_score
            best_tau = tau

    print(f"Optimal threshold found: tau* = {best_tau:.2f} (Macro F_0.5 = {best_f05:.4f})")

    # Save model artifact
    model_path = os.path.join(artifacts_dir, 'model.joblib')
    save_model(clf, model_path)
    print(f"Saved model to {model_path}.")

    return clf, best_tau


def run_inference_partition(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    clf: object,
    threshold: float,
    country_name: str,
    artifacts_dir: str,
    top_k: int = 20,
    batch_size: int = 50000
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Runs candidate blocking, feature extraction, and matching inference
    for a single country partition with checkpointing.
    Returns (candidate_map, matched_map).
    """
    cand_chk_path = os.path.join(artifacts_dir, f"chk_cand_{country_name}.parquet")
    match_chk_path = os.path.join(artifacts_dir, f"chk_match_{country_name}.parquet")

    # Check if partition already checkpointed
    if os.path.isfile(cand_chk_path) and os.path.isfile(match_chk_path):
        print(f"\n[Checkpoint] Found existing predictions for [{country_name}], loading from disk...")
        df_cand = pl.read_parquet(cand_chk_path)
        df_match = pl.read_parquet(match_chk_path)
        cand_map = {r['source1_entity_id']: r['candidate_entity_ids'].split(',') if r['candidate_entity_ids'] else [] for r in df_cand.to_dicts()}
        match_map = {r['source1_entity_id']: r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else [] for r in df_match.to_dicts()}
        return cand_map, match_map

    print(f"\nProcessing Country Partition: [{country_name}] (S1: {len(s1_df)}, S2: {len(s2_df)}, S3: {len(s3_df)})...")
    t0 = time.time()

    # Combine targets
    targets = pl.concat([s2_df, s3_df]).to_dicts()
    print(f"  Target pool: {len(targets)} records.")

    # Fit blocker
    blocker = MultiKeyBlocker(max_postings_token=1000, max_postings_digit=500, top_k=top_k)
    blocker.fit(targets)
    print(f"  Blocker indexed in {time.time() - t0:.2f}s.")

    s1_records = s1_df.to_dicts()
    all_candidates: Dict[str, List[str]] = {}
    predictions_by_s1: Dict[str, List[Tuple[str, float]]] = {}

    t_inf = time.time()
    # Process S1 entities in batches
    for b_start in range(0, len(s1_records), batch_size):
        b_end = min(b_start + batch_size, len(s1_records))
        batch = s1_records[b_start:b_end]

        X_batch, meta_batch = [], []
        for r in batch:
            s1_id = r['entity_id']
            s1_cache = {
                'norm_name': normalize_text(r['business_name']),
                'norm_addr': normalize_text(r['business_address']),
                'digits': extract_digits(normalize_text(r['business_address'])),
                'tokens': set(tokenize(normalize_text(r['business_name']), min_len=2)),
                'ngrams': get_char_ngrams(normalize_text(r['business_name']), n=3)
            }

            cands = blocker.retrieve_candidates(r['business_name'], r['business_address'])
            cand_tids = [blocker.target_metadata[tidx]['entity_id'] for tidx, _ in cands]
            all_candidates[s1_id] = cand_tids

            for tidx, b_score in cands:
                t_meta = blocker.target_metadata[tidx]
                feat = compute_pair_features(s1_cache, t_meta, b_score)
                X_batch.append(feat)
                meta_batch.append((s1_id, t_meta['entity_id']))

        if X_batch:
            X_mat = np.array(X_batch, dtype=np.float32)
            probs = clf.predict_proba(X_mat)[:, 1]
            for (s1_id, tid), p in zip(meta_batch, probs):
                if s1_id not in predictions_by_s1:
                    predictions_by_s1[s1_id] = []
                predictions_by_s1[s1_id].append((tid, float(p)))
        else:
            for r in batch:
                all_candidates[r['entity_id']] = []

        print(f"  Processed {b_end}/{len(s1_records)} entities (elapsed: {time.time() - t_inf:.1f}s)...")

    # Post-processing: Singleton gate & Injectivity conflict resolution
    print(f"  Applying single-parent conflict resolution (threshold={threshold:.2f})...")
    matched_map = resolve_matches_and_conflicts(predictions_by_s1, threshold=threshold, singleton_gate=0.35)

    # Save checkpoint
    chk_cands_list = [{'source1_entity_id': s1_id, 'candidate_entity_ids': ','.join(cands)} for s1_id, cands in all_candidates.items()]
    chk_match_list = [{'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(matched_map.get(s1_id, []))} for s1_id in all_candidates.keys()]
    pl.DataFrame(chk_cands_list).write_parquet(cand_chk_path)
    pl.DataFrame(chk_match_list).write_parquet(match_chk_path)
    print(f"  Checkpointed [{country_name}] results to disk.")

    return all_candidates, matched_map


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Master Pipeline")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Path to training data directory")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test data directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Path to model artifacts directory")
    parser.add_argument("--sample-train", type=int, default=30000, help="Number of training entities to sample")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K candidates per entity in blocking")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (if None, optimized from train)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    # Resolve paths relative to working directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # 1. Model Loading / Training
    model_path = os.path.join(args.artifacts_dir, 'model.joblib')
    if os.path.isfile(model_path) and not args.retrain:
        print(f"\n[1/6] Found existing trained model artifact at {model_path}. Loading...")
        clf = load_model(model_path)
        opt_threshold = 0.60
    else:
        clf, opt_threshold = train_model_pipeline(
            train_dir=args.train_dir,
            artifacts_dir=args.artifacts_dir,
            sample_entities=args.sample_train,
            top_k=args.top_k
        )
    final_threshold = args.threshold if args.threshold is not None else opt_threshold

    # 2. Test Set Inference by Country Partition
    print("\n[2/6] Loading Test Datasets by Country Partition...")
    test_s1_lazy = pl.scan_csv(os.path.join(args.test_dir, 'test_source1.tsv'), separator='\t')
    test_s2_lazy = pl.scan_csv(os.path.join(args.test_dir, 'test_source2.tsv'), separator='\t')
    test_s3_lazy = pl.scan_csv(os.path.join(args.test_dir, 'test_source3.tsv'), separator='\t')

    # Read required S1 order
    s1_all_ids = test_s1_lazy.select('entity_id').collect()['entity_id'].to_list()
    print(f"Total reference entities to resolve: {len(s1_all_ids):,}")

    countries = ['France', 'US', 'India']
    all_test_candidates: Dict[str, List[str]] = {}
    all_test_matches: Dict[str, List[str]] = {}

    for c in countries:
        s1_c = test_s1_lazy.filter(pl.col('country') == c).collect()
        s2_c = test_s2_lazy.filter(pl.col('country') == c).collect()
        s3_c = test_s3_lazy.filter(pl.col('country') == c).collect()

        cand_map, match_map = run_inference_partition(
            s1_df=s1_c,
            s2_df=s2_c,
            s3_df=s3_c,
            clf=clf,
            threshold=final_threshold,
            country_name=c,
            artifacts_dir=args.artifacts_dir,
            top_k=args.top_k
        )
        all_test_candidates.update(cand_map)
        all_test_matches.update(match_map)

    # 3. Write Output Files
    cand_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    match_file = os.path.join(args.output_dir, "matching_results.tsv")

    print(f"\n[3/6] Writing {cand_file} and {match_file}...")
    with open(cand_file, "w", encoding="utf-8") as f_cand, open(match_file, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for s1_id in s1_all_ids:
            cands = all_test_candidates.get(s1_id, [])
            matches = all_test_matches.get(s1_id, [])

            # Guaranteed subset check: matches must be subset of cands
            cand_set = set(cands)
            valid_matches = [m for m in matches if m in cand_set]

            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            f_match.write(f"{s1_id}\t{','.join(valid_matches)}\n")

    print(f"Successfully written {len(s1_all_ids):,} rows to both output files.")

    # 4. Local Validation
    validator_candidates = [
        os.path.normpath(os.path.join(args.test_dir, "..", "..", "utils", "validate_submission.py")),
        "utils/validate_submission.py",
        "student_resource/utils/validate_submission.py"
    ]
    validator_path = next((p for p in validator_candidates if os.path.isfile(p)), None)
    if validator_path:
        print(f"\n[4/6] Running automated local submission validation with {validator_path}...")
        cmd = f"python3 {validator_path} --matching {match_file} --candidate {cand_file} --test-dir {args.test_dir}"
        ret = os.system(cmd)
        if ret == 0:
            print("\n*** SUBMISSION VALIDATION PASSED (EXIT CODE 0) ***")
        else:
            print(f"\nValidation failed with exit code {ret}.")
    else:
        print("Validator script not found, skipped local validation check.")


if __name__ == "__main__":
    main()
