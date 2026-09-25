# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CodeNova  
**Team Members:** Harsh  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary

We developed an end-to-end, multi-stage machine learning system for enterprise-scale Business Entity Resolution across 26.4 million records spanning US, India, and France. Our approach integrates domain-agnostic Unicode NFKD normalization, multi-key inverted index blocking ($\ge 98.2\%$ recall), a 16-dimensional RapidFuzz pairwise feature engine, and a precision-calibrated LightGBM gradient boosted matcher. By exploiting the mathematical Single-Parent (Injectivity) Invariant of the reference dataset and optimizing decision boundaries directly against the $4\times$ False Positive penalty of the Macro-$F_{0.5}$ metric, our solution achieves an empirical validation **Macro-$F_{0.5}$ score of 0.9693** with linear scaling on standard workstation hardware.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 26,436,001 total records revealed critical structural and mathematical invariants:
1. **The Country Invariance Theorem:** Across 182,932 ground-truth match pairs analyzed in the training set, $0$ cross-country matches exist ($P(\text{country}(s_1) \neq \text{country}(m) \mid m \in \text{match}(s_1)) = 0$). This allows strict country-level hard partitioning ($\mathcal{S}_{\text{France}}, \mathcal{S}_{\text{US}}, \mathcal{S}_{\text{India}}$), reducing search complexity by $\sim 65\%$ and eliminating peak memory overhead.
2. **The Single-Parent (Injectivity) Invariant:** Across all 7,638,365 matched target records in ground truth, exactly $0$ target entities from $\mathcal{S}_2$ or $\mathcal{S}_3$ map to multiple $\mathcal{S}_1$ reference entities. The ground truth induces a partial injection:
   $$f: (\mathcal{S}_2 \cup \mathcal{S}_3) \to \mathcal{S}_1 \cup \{\bot\}$$
3. **Singleton Dynamics:** $\sim 5.59\%$ of reference records in $\mathcal{S}_1$ have zero matching records ($Y(s_1) = \emptyset$). Under Macro-$F_{0.5}$, falsely predicting any candidate for a true singleton causes a catastrophic drop from $1.0 \to 0.0$.
4. **Out-of-Distribution Language Shift (France):** France accounts for 259,452 reference entities (15.0%) in the test set but is absent from training. This necessitated language-agnostic character n-grams and universal corporate suffix canonicalization (`SARL`, `SAS`, `EURL`, `SA`, `SCI`).
5. **Noise & Missing Attribute Patterns:** Downstream sources exhibit ~3.3% missing address fields in $\mathcal{S}_2/\mathcal{S}_3$, brand name / DBA variations (e.g. `Dréxkor` matching `Maure Williams Colombier Inc`), Indic script transliterations (Devanagari, Tamil), and domain name strings (`maurewilliamscolombier.com`).

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi-Key Inverted Index Blocking + Pairwise Gradient Boosted Matcher + Global Conflict Resolution  
**Core Innovation:** A two-tier mathematical optimization pipeline combining:
1. **Multi-pass Inverted Indexing:** Coupling significant name tokens, address building/PIN numeric tokens, and locality n-grams to guarantee high recall ($\ge 98.2\%$) while bounding candidate comparisons to $K \le 20$ per entity.
2. **Injectivity-Constrained Post-Processing:** Enforcing the Single-Parent Ground-Truth Invariant via greedy maximum-margin conflict resolution: $\hat{s}_1(m) = \arg\max_{s_1} P(s_1, m)$, which mathematically eliminates false positive multi-merges and protects the precision-heavy metric.

---

## 3. Candidate Generation (Blocking)

To reduce the $2.2 \times 10^{13}$ pairwise comparison space down to a scalable candidate pool:

- **Blocking keys used:**
  1. **Primary Hard Partition:** Exact Country Matching ($\text{Country}(s_1) = \text{Country}(t)$).
  2. **Pass A (Name Core Tokens):** Inverted index on clean word tokens (length $\ge 3$) with upper-frequency posting caps to prune non-discriminative words.
  3. **Pass B (Address Numeric Anchors):** Inverted index on extracted building numbers, plot IDs, and PIN/postal codes.
  4. **Pass C (Locality & Street Tokens):** Inverted index on distinctive locality tokens (length $\ge 4$).
- **Candidate pairs generated:** Average $16.8$ candidates per entity ($28.7\text{M}$ candidate pairs across $1,732,544$ test entities).
- **How true matches were preserved:** Disjunctive multi-index union ensures that records with transliterated/DBA names are retrieved via address numeric anchors, while records with missing addresses are retrieved via core name tokens. Empirically, this achieved a **98.22% Pairs Completeness (Recall)** ceiling on validation benchmarks.

---

## 4. Matching Model

**Features used (16 Dimensions):**
- **Name Similarity Features:**
  - Jaro-Winkler distance: $JW(\text{name}_1, \text{name}_2)$
  - Normalized Levenshtein ratio: `fuzz.ratio`
  - Token Sort Ratio & Token Set Ratio (order-invariant token matching)
  - Partial string matching ratio: `fuzz.partial_ratio`
  - Character 3-gram Jaccard similarity: $\frac{|G_3(n_1) \cap G_3(n_2)|}{|G_3(n_1) \cup G_3(n_2)|}$
  - Word token Jaccard similarity: $\frac{|T(n_1) \cap T(n_2)|}{|T(n_1) \cup T(n_2)|}$
  - Length absolute difference: $|len(n_1) - len(n_2)|$
- **Address Similarity Features:**
  - Normalized Address Levenshtein ratio
  - Address Token Sort Ratio & Token Set Ratio
  - Numeric Digits Jaccard similarity: $\frac{|\text{digits}_1 \cap \text{digits}_2|}{|\text{digits}_1 \cup \text{digits}_2|}$
  - Exact Digits Match indicator: $\mathbb{I}(\text{digits}_1 = \text{digits}_2 \neq \emptyset)$
  - Address missing indicator: $\mathbb{I}(\text{address}_t \text{ is None})$
- **Meta & Interaction Features:**
  - Source flag: $\mathbb{I}(t \in \mathcal{S}_2)$ vs $\mathbb{I}(t \in \mathcal{S}_3)$
  - Initial lexical blocking score

**Model type:** LightGBM Gradient Boosted Decision Tree (`n_estimators=180, learning_rate=0.08, num_leaves=31`).  
**Threshold selection method:** Grid search on held-out validation cluster split specifically maximizing Macro-$F_{0.5}$. The metric's $4\times$ penalty on False Positives ($F_{0.5} = \frac{5 \cdot TP}{|Y| + 4 \cdot |\hat{Y}|}$) yielded an optimal conservative decision threshold $\tau^* = 0.60$ with a singleton gate cutoff at $0.35$.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9693** on held-out entity validation fold.
- **Common false positives (wrong merges):** Co-located businesses at identical commercial complexes (e.g. distinct shops in the same shopping mall or business park sharing building number and street address) where name similarity was borderline. Mitigated by setting a high precision threshold ($\tau^* = 0.60$) and requiring strong name subword agreement when address is shared.
- **Common false negatives (missed matches):** Extreme cross-lingual transliterations where the target record contained both a transliterated Indic script name and a missing/garbled street number.

---

## 6. Conclusion

By grounding the solution in verified mathematical invariants (country partitioning and single-parent injectivity) and coupling high-recall multi-key inverted indexing with precision-tuned LightGBM scoring, we resolved 1.73 million multi-lingual test entities in minutes on CPU hardware without external lookups. The resulting pipeline generates strictly validated submissions achieving **0.9693 Macro-$F_{0.5}$** validation performance while fully generalizing to unobserved international markets.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized as follows:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored leaderboard output (1,732,544 rows)
│   └── candidate_pairs.tsv         # Blocking candidate set (1,732,544 rows)
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalizer.py       # Domain-agnostic NFKD text & suffix cleaner
│       │   ├── blocking.py         # Multi-key inverted index candidate generator
│       │   ├── features.py         # 16-D RapidFuzz pairwise feature engine
│       │   ├── model.py            # LightGBM trainer & model serializer
│       │   ├── postprocess.py      # Injectivity conflict solver & singleton gate
│       │   └── run_pipeline.py     # Master end-to-end pipeline runner
│       ├── artifacts/
│       │   └── model.joblib        # Trained LightGBM model weights
│       ├── README.md               # End-to-end reproduction guide
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Technical methodology writeup
```

**Single Command Reproduction:**
```bash
python3 code/business_entity_resolution/src/run_pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --artifacts-dir code/business_entity_resolution/artifacts \
    --top-k 20
```

### B. Validation Results
Automated validation run via `utils/validate_submission.py`:
- `test_source1.tsv` reference entities: 1,732,544
- `matching_results.tsv`: 1,732,544 rows (538,411 singletons, 1,194,133 matched)
- `candidate_pairs.tsv`: 1,732,544 rows (246,716 empty, 1,485,828 non-empty)
- **Validator Status:** `PASS — no blocking issues found. Safe to submit. (Exit code 0)`
