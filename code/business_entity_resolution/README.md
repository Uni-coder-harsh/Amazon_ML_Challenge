# Business Entity Resolution Pipeline
### Amazon ML Challenge 2026

A high-throughput, multi-source business entity resolution system designed to resolve noisy commercial records across three disparate sources ($S_1, S_2, S_3$) under precision-heavy evaluation ($F_{0.5}$).

---

### 1. Architecture Overview

1. **Country-Partitioned Ingestion:** Exploits empirical 100% country invariance ($\text{Country}(S_1) = \text{Country}(S_2/S_3)$) to process France, US, and India independently, bounding memory usage under 4 GB.
2. **Domain-Agnostic Normalizer:** Decomposes Unicode (NFKD accent-folding), canonicalizes multi-jurisdictional legal forms (US LLC/Inc, India Pvt Ltd/LLP, France SARL/SAS/EURL), extracts numeric address tokens, and folds subwords.
3. **Multi-Key Inverted Index Blocking:** Generates candidate pairs with $\ge 98\%$ Pairs Completeness (Recall) via multi-pass inverted indexing on significant name tokens, address numbers, and locality tokens.
4. **16-Dimensional RapidFuzz Feature Engine:** Vectorizes name token/partial/sort/set ratios, Jaro-Winkler distances, character 3-gram Jaccard, address edit distances, numeric digit Jaccard, and missing address indicators at $>100,000$ pairs/second.
5. **Precision-Tuned LightGBM Matcher:** Gradient Boosted Decision Tree optimized specifically for the $4\times$ False Positive penalty of the $F_{0.5}$ metric ($\tau^* \approx 0.70$).
6. **Injectivity & Singleton Post-Processing:** Enforces the Single-Parent Ground-Truth Invariant via greedy maximum-margin conflict resolution and guards singletons with a confidence cutoff gate.

---

### 2. Environment Setup

```bash
# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 3. End-to-End Reproduction Instructions

To execute the entire pipeline from scratch (training $\to$ blocking $\to$ feature extraction $\to$ inference $\to$ output generation $\to$ verification):

From the `student_resource/` directory:

```bash
python3 code/business_entity_resolution/src/run_pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --artifacts-dir code/business_entity_resolution/artifacts \
    --sample-train 30000 \
    --top-k 20
```

Outputs produced in `output/`:
- `output/matching_results.tsv` — Final entity matches scored on leaderboard.
- `output/candidate_pairs.tsv` — Blocking candidate set fed to model.

---

### 4. Validation

Validate formatting and constraints before uploading:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
