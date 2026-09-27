"""
run_pipeline.py  ─  v2.2 (smart training + chunked inference)
=============================================================

Key optimizations vs v2.1:
  • Training uses only 50k S1 entities (+ their true matches + sampled negatives)
    → No need to embed all 10M training targets
  • Inference encodes targets once per partition, cached to disk
  • Embedding step uses show_progress_bar=True for monitoring
  • Chunked inference in batches of 50k S1 to control memory

Architecture:
  TRAIN:
    Sample 50k S1 from each country (proportional)
    Build lexical index over ALL targets (fast, in-memory)
    For training pairs: true positives (from GT) + lexical-blocked negatives
    Embed only the ~300K candidate pairs' texts (not all 10M targets)
    Train XGBoost

  TEST:
    Per country: encode ALL targets once → FAISS index
    Batch S1 queries through lexical + ANN → candidates → features → classify
"""

from __future__ import annotations
import os, gc, sys, time, joblib, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE       = Path(__file__).resolve().parents[3]
TRAIN_DIR  = BASE / "dataset" / "train"
TEST_DIR   = BASE / "dataset" / "test"
OUTPUT_DIR = BASE / "output"
ART_DIR    = BASE / "code" / "business_entity_resolution" / "artifacts"
SRC_DIR    = Path(__file__).parent

OUTPUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from normalizer import normalize, get_tokens, extract_digits, get_ngrams
from blocking import LexicalBlocker
from features import compute_features, N_FEATURES

# ── Config ─────────────────────────────────────────────────────────────────
USE_EMBEDDINGS    = True
TOP_K_ANN         = 15
MAX_CANDIDATES    = 40          # smaller = faster, better ranking score
EMBED_BATCH       = 512
EMB_SCORE_THRESH  = 0.50
TRAIN_S1_SAMPLE   = 50_000     # per country, for training
INFER_S1_CHUNK    = 20_000     # batch size for inference
MODEL_NAME        = "paraphrase-multilingual-mpnet-base-v2"

print("="*70)
print("Pipeline v2.2 — Smart Training + Chunked Inference")
print("="*70)


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    print("\n[1] Loading data...")
    tr1 = pd.read_csv(TRAIN_DIR/"train_source1.tsv", sep="\t", dtype=str).fillna("")
    tr2 = pd.read_csv(TRAIN_DIR/"train_source2.tsv", sep="\t", dtype=str).fillna("")
    tr3 = pd.read_csv(TRAIN_DIR/"train_source3.tsv", sep="\t", dtype=str).fillna("")
    gt  = pd.read_csv(TRAIN_DIR/"train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    te1 = pd.read_csv(TEST_DIR/"test_source1.tsv", sep="\t", dtype=str).fillna("")
    te2 = pd.read_csv(TEST_DIR/"test_source2.tsv", sep="\t", dtype=str).fillna("")
    te3 = pd.read_csv(TEST_DIR/"test_source3.tsv", sep="\t", dtype=str).fillna("")

    gt['match_list'] = gt['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    print(f"  Train S1={len(tr1):,} S2={len(tr2):,} S3={len(tr3):,}")
    print(f"  Test  S1={len(te1):,} S2={len(te2):,} S3={len(te3):,}")
    print(f"  Test countries: {sorted(te1['country'].unique())}")
    return tr1, tr2, tr3, gt, te1, te2, te3


def to_lkp(df: pd.DataFrame) -> dict:
    return {r.entity_id: {"name": r.business_name, "addr": r.business_address, "country": r.country}
            for r in df.itertuples(index=False)}


def make_text(name: str, addr: str) -> str:
    parts = []
    if name and name.strip(): parts.append(name.strip())
    if addr and addr.strip():  parts.append(addr.strip()[:80])
    return " | ".join(parts) if parts else "unknown"


def f05(prec, rec):
    return (1.25 * prec * rec) / max(0.25 * prec + rec, 1e-9)


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ════════════════════════════════════════════════════════════════════════════

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [emb] Loading {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = 128
    return _model


def encode(texts: list[str], desc: str = "") -> np.ndarray:
    m = get_model()
    if desc:
        print(f"  [emb] Encoding {len(texts):,} texts ({desc})...")
    return m.encode(texts, batch_size=EMBED_BATCH,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True).astype(np.float32)


def build_faiss_index(embs: np.ndarray):
    import faiss
    n, d = embs.shape
    if n < 5000:
        idx = faiss.IndexFlatIP(d)
    else:
        nlist = min(256, max(8, n // 100))
        q = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(embs)
        idx.nprobe = min(32, nlist)
    idx.add(embs)
    return idx


def ann_query(q_embs: np.ndarray, idx, t_ids: list[str],
              top_k: int) -> dict[int, list[str]]:
    k = min(top_k, len(t_ids))
    scores, indices = idx.search(q_embs, k)
    result = {}
    for qi in range(len(q_embs)):
        cands = []
        for j, s in zip(indices[qi], scores[qi]):
            if j >= 0 and s >= EMB_SCORE_THRESH:
                cands.append(t_ids[j])
        result[qi] = cands
    return result


# ════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def build_pair_features(pairs: list[tuple], s1_lkp: dict, t_lkp: dict,
                        s1_embs: dict | None, t_embs: dict | None) -> np.ndarray:
    """pairs: list of (s1_id, cand_id)"""
    rows = []
    for s1_id, cid in pairs:
        s1 = s1_lkp.get(s1_id, {})
        cd = t_lkp.get(cid, {})
        ea = s1_embs.get(s1_id) if s1_embs else None
        eb = t_embs.get(cid)    if t_embs  else None
        feat = compute_features(
            s1.get("name",""), s1.get("addr",""),
            cd.get("name",""), cd.get("addr",""),
            emb_a=ea, emb_b=eb,
            country_a=s1.get("country",""), country_b=cd.get("country",""),
        )
        rows.append(feat)
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, N_FEATURES), np.float32)


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_model(tr1, tr2, tr3, gt):
    print("\n[3] Training classifier...")
    from sklearn.model_selection import train_test_split

    gt_dict = {r.source1_entity_id: set(r.match_list) for r in gt.itertuples(index=False)}
    s1_lkp  = to_lkp(tr1)
    s23_lkp = {**to_lkp(tr2), **to_lkp(tr3)}
    tgt_df  = pd.concat([tr2, tr3], ignore_index=True)

    all_pairs, all_X, all_y = [], [], []

    for country in sorted(tr1['country'].unique()):
        s1_part  = tr1[tr1['country'] == country]
        tgt_part = tgt_df[tgt_df['country'] == country]
        if len(s1_part) == 0 or len(tgt_part) == 0: continue

        # Sample S1 for training
        n_samp = min(TRAIN_S1_SAMPLE, len(s1_part))
        s1_samp = s1_part.sample(n=n_samp, random_state=42)
        print(f"\n  [train:{country}] S1_sample={n_samp:,} targets={len(tgt_part):,}")

        # ── Lexical blocking ──────────────────────────────────────────────
        t0 = time.time()
        blocker = LexicalBlocker(max_candidates=MAX_CANDIDATES)
        blocker.index([(r.entity_id, r.business_name, r.business_address)
                       for r in tgt_part.itertuples(index=False)])
        print(f"  [lex:{country}] Index built in {time.time()-t0:.0f}s")

        pairs: list[tuple[str,str]] = []
        for r in s1_samp.itertuples(index=False):
            cands = blocker.query(r.business_name, r.business_address, use_ngrams=True)
            true_m = gt_dict.get(r.entity_id, set())
            # Always include true positives even if not in candidates
            for tm in true_m:
                if tm in s23_lkp and s23_lkp[tm]['country'] == country:
                    if tm not in cands:
                        cands.add(tm)
            for c in cands:
                pairs.append((r.entity_id, c))

        y = np.array([1 if cid in gt_dict.get(sid, set()) else 0
                      for sid, cid in pairs], dtype=np.int32)
        pos = y.sum(); neg = len(y) - pos
        print(f"  [train:{country}] pairs={len(pairs):,} pos={pos:,} neg={neg:,}")

        # ── Embeddings for pair texts only ────────────────────────────────
        s1_emb_dict, t_emb_dict = None, None
        if USE_EMBEDDINGS:
            # Collect unique IDs in pairs
            unique_s1 = list({sid for sid, _ in pairs})
            unique_t  = list({cid for _, cid in pairs})

            s1_texts = [make_text(s1_lkp[x]["name"], s1_lkp[x]["addr"]) for x in unique_s1]
            t_texts  = [make_text(s23_lkp[x]["name"], s23_lkp[x]["addr"]) for x in unique_t if x in s23_lkp]
            unique_t_valid = [x for x in unique_t if x in s23_lkp]

            s1_vecs = encode(s1_texts, desc=f"S1 ({country})")
            t_vecs  = encode(t_texts,  desc=f"targets ({country})")

            s1_emb_dict = {uid: s1_vecs[i] for i, uid in enumerate(unique_s1)}
            t_emb_dict  = {uid: t_vecs[i]  for i, uid in enumerate(unique_t_valid)}
            del s1_vecs, t_vecs; gc.collect()

        X = build_pair_features(pairs, s1_lkp, s23_lkp, s1_emb_dict, t_emb_dict)
        all_pairs.extend(pairs)
        all_X.append(X)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    pos = y_all.sum(); neg = len(y_all) - pos
    print(f"\n  Total pairs: {len(y_all):,}  pos={pos:,} ({pos/len(y_all)*100:.1f}%)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)

    scale_pw = neg / max(pos, 1)
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale_pw,
                            eval_metric="logloss", tree_method="hist",
                            n_jobs=-1, random_state=42, verbosity=0)
        print("  Using XGBoost")
    except ImportError:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
        print("  Using LightGBM fallback")

    clf.fit(X_tr, y_tr)

    # Threshold tuning on val
    val_prob = clf.predict_proba(X_val)[:, 1]
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (val_prob >= t).astype(int)
        tp = int(((pred==1)&(y_val==1)).sum())
        fp = int(((pred==1)&(y_val==0)).sum())
        fn = int(((pred==0)&(y_val==1)).sum())
        p = tp / max(tp+fp, 1); r = tp / max(tp+fn, 1)
        f = f05(p, r)
        if f > best_f: best_f, best_t = f, t

    print(f"  Best threshold={best_t:.2f}  val-F0.5={best_f:.4f}")
    clf.fit(X_all, y_all)  # retrain on full

    path = ART_DIR / "model_v2.joblib"
    joblib.dump({"model": clf, "threshold": best_t}, path)
    print(f"  Model saved → {path}")
    return clf, best_t


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE — chunked per country
# ════════════════════════════════════════════════════════════════════════════

def infer_country(s1_df: pd.DataFrame, tgt_df: pd.DataFrame,
                  country: str, clf, threshold: float,
                  s1_lkp: dict, t_lkp: dict) -> tuple[dict, dict]:
    """Returns (candidates, matches) for this country partition."""
    print(f"\n  [infer:{country}] S1={len(s1_df):,} targets={len(tgt_df):,}")
    cands_out:  dict[str,set[str]] = {}
    matches_out: dict[str,set[str]] = {}

    if len(tgt_df) == 0:
        for eid in s1_df['entity_id']: cands_out[eid] = set(); matches_out[eid] = set()
        return cands_out, matches_out

    # ── Lexical index (built once for this country) ────────────────────────
    t0 = time.time()
    blocker = LexicalBlocker(max_candidates=MAX_CANDIDATES)
    blocker.index([(r.entity_id, r.business_name, r.business_address)
                   for r in tgt_df.itertuples(index=False)])
    print(f"  [lex:{country}] Index built in {time.time()-t0:.0f}s")

    # ── Embedding index (built once for this country) ──────────────────────
    faiss_idx, t_ids = None, []
    if USE_EMBEDDINGS:
        t_ids   = list(tgt_df['entity_id'])
        t_texts = [make_text(r.business_name, r.business_address)
                   for r in tgt_df.itertuples(index=False)]
        t_vecs  = encode(t_texts, desc=f"targets-{country}")
        faiss_idx = build_faiss_index(t_vecs)
        del t_vecs; gc.collect()
        print(f"  [emb:{country}] FAISS index built ({len(t_ids):,} vectors)")

    # ── Process S1 in chunks ───────────────────────────────────────────────
    s1_ids = list(s1_df['entity_id'])
    for chunk_start in range(0, len(s1_ids), INFER_S1_CHUNK):
        chunk_ids  = s1_ids[chunk_start : chunk_start + INFER_S1_CHUNK]
        chunk_df   = s1_df.iloc[chunk_start : chunk_start + INFER_S1_CHUNK]

        # Lexical candidates
        chunk_cands: dict[str, set[str]] = {}
        for r in chunk_df.itertuples(index=False):
            chunk_cands[r.entity_id] = blocker.query(
                r.business_name, r.business_address, use_ngrams=True)

        # ANN candidates
        if USE_EMBEDDINGS and faiss_idx is not None:
            q_texts = [make_text(r.business_name, r.business_address)
                       for r in chunk_df.itertuples(index=False)]
            q_vecs  = encode(q_texts)
            ann_res = ann_query(q_vecs, faiss_idx, t_ids, top_k=TOP_K_ANN)
            del q_vecs; gc.collect()
            for qi, qid in enumerate(chunk_ids):
                for tid in ann_res.get(qi, []):
                    chunk_cands[qid].add(tid)

        # Cap candidates
        for qid in chunk_cands:
            if len(chunk_cands[qid]) > MAX_CANDIDATES:
                chunk_cands[qid] = set(list(chunk_cands[qid])[:MAX_CANDIDATES])

        # Build pairs & features
        pairs = [(sid, cid) for sid, cids in chunk_cands.items() for cid in cids]
        if pairs:
            X = build_pair_features(pairs, s1_lkp, t_lkp, None, None)
            probs = clf.predict_proba(X)[:, 1]
        else:
            probs = np.array([])

        # Collect results
        prob_iter = iter(probs)
        for sid, cids in chunk_cands.items():
            cands_out[sid]  = cids
            matches_out[sid] = set()
            for cid in cids:
                p = next(prob_iter)
                if p >= threshold:
                    matches_out[sid].add(cid)

        n_done    = min(chunk_start + INFER_S1_CHUNK, len(s1_ids))
        n_matched = sum(1 for v in matches_out.values() if v)
        print(f"  [{country}] {n_done:,}/{len(s1_ids):,} S1 done | matched={n_matched:,}")

    return cands_out, matches_out


def run_inference(te1, te2, te3, clf, threshold):
    print("\n[4] Running inference on test data...")
    s1_lkp = to_lkp(te1)
    t_lkp  = {**to_lkp(te2), **to_lkp(te3)}
    tgt_df = pd.concat([te2, te3], ignore_index=True)

    all_cands:   dict[str,set[str]] = {eid: set() for eid in te1['entity_id']}
    all_matches: dict[str,set[str]] = {eid: set() for eid in te1['entity_id']}

    for country in sorted(te1['country'].unique()):
        s1_part  = te1[te1['country'] == country]
        tgt_part = tgt_df[tgt_df['country'] == country]
        c_cands, c_matches = infer_country(
            s1_part, tgt_part, country, clf, threshold, s1_lkp, t_lkp)
        all_cands.update(c_cands)
        all_matches.update(c_matches)

    return all_cands, all_matches


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def write_outputs(te1, cands, matches):
    print("\n[5] Writing outputs...")
    mrows, crows = [], []
    for eid in te1['entity_id']:
        m = sorted(matches.get(eid, set()))
        c = sorted(cands.get(eid, set()))
        mrows.append({"source1_entity_id": eid, "matched_entity_ids":    ",".join(m)})
        crows.append({"source1_entity_id": eid, "candidate_entity_ids": ",".join(c)})

    pd.DataFrame(mrows).to_csv(OUTPUT_DIR/"matching_results.tsv",  sep="\t", index=False)
    pd.DataFrame(crows).to_csv(OUTPUT_DIR/"candidate_pairs.tsv",   sep="\t", index=False)

    n_m = sum(1 for r in mrows if r["matched_entity_ids"])
    n_s = len(mrows) - n_m
    tot_c = sum(len(r["candidate_entity_ids"].split(",")) if r["candidate_entity_ids"] else 0 for r in crows)
    print(f"  matching_results : matched={n_m:,} singletons={n_s:,} ({n_s/len(mrows)*100:.1f}%)")
    print(f"  candidate_pairs  : total={tot_c:,} avg/S1={tot_c/max(len(crows),1):.1f}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    tr1, tr2, tr3, gt, te1, te2, te3 = load_data()

    model_path = ART_DIR / "model_v2.joblib"
    if model_path.exists():
        print(f"\n[2/3] Loading cached model: {model_path}")
        art = joblib.load(model_path)
        clf, threshold = art["model"], art["threshold"]
        print(f"  threshold={threshold:.2f}")
    else:
        clf, threshold = train_model(tr1, tr2, tr3, gt)

    cands, matches = run_inference(te1, te2, te3, clf, threshold)
    write_outputs(te1, cands, matches)

    print(f"\n✅ DONE — total time: {(time.time()-t0)/60:.1f} min")
    print("Validate: python utils/validate_submission.py output/")


if __name__ == "__main__":
    main()
