# Product Requirements Document (PRD)
## Project: Amazon ML Challenge 2026 — Business Entity Resolution (ER)

---

### 1. Business & Problem Context

In global e-commerce and commercial catalog platforms like Amazon, merchant, vendor, brand, and supplier records are continuously ingested from thousands of disparate, external, and semi-structured feeds. These feeds frequently lack standardized unique business identifiers (e.g., DUNS, EIN, GSTIN, or SIREN numbers). Instead, records contain noisy, fragmented, abbreviated, and country-specific representations of business identities.

The **Business Entity Resolution (ER)** task requires identifying which heterogeneous records across three distinct sources ($\mathcal{S}_1, \mathcal{S}_2, \mathcal{S}_3$) refer to the exact same real-world commercial entity. 

- **Source 1 ($\mathcal{S}_1$):** Serves as the deduplicated reference ("golden record") catalogue.
- **Source 2 ($\mathcal{S}_2$) & Source 3 ($\mathcal{S}_3$):** Noisy downstream ingestion sources with duplicate, partial, transliterated, or missing attributes.

The objective is to produce:
1. `candidate_pairs.tsv`: The high-recall candidate set generated during blocking.
2. `matching_results.tsv`: The high-precision predicted links from $\mathcal{S}_1$ into $(\mathcal{S}_2 \cup \mathcal{S}_3)$.

---

### 2. Mathematical Problem Formulation

#### 2.1 Entity Universe & Set Relations
Let $\mathcal{E}$ denote the universe of real-world business entities.
Let the input records be partitioned into three disjoint sets:
$$\mathcal{S}_1 = \{s_1^{(1)}, s_1^{(2)}, \dots, s_1^{(N_1)}\}$$
$$\mathcal{S}_2 = \{s_2^{(1)}, s_2^{(2)}, \dots, s_2^{(N_2)}\}$$
$$\mathcal{S}_3 = \{s_3^{(1)}, s_3^{(2)}, \dots, s_3^{(N_3)}\}$$

Each record $r \in \mathcal{S}_1 \cup \mathcal{S}_2 \cup \mathcal{S}_3$ is a tuple:
$$r = (\text{id}(r), \text{name}(r), \text{address}(r), \text{country}(r))$$

There exists a ground-truth mapping function $\phi: (\mathcal{S}_1 \cup \mathcal{S}_2 \cup \mathcal{S}_3) \to \mathcal{E}$ mapping each record to its real-world business entity.

#### 2.2 Golden Source & Injectivity Invariants
From empirical data verification across all 2,206,821 training ground-truth rows:
1. **Deduplicated Reference:** $\forall u, v \in \mathcal{S}_1, u \neq v \implies \phi(u) \neq \phi(v)$. Each entity in $\mathcal{S}_1$ represents a unique cluster head.
2. **Injectivity of Matching (Single Parent Invariant):** 
   $$\forall m \in (\mathcal{S}_2 \cup \mathcal{S}_3), \quad |\{s_1 \in \mathcal{S}_1 : \phi(s_1) = \phi(m)\}| \le 1$$
   *Proof from training ground truth:* Across 7,638,365 matched IDs in train, **0** IDs from $\mathcal{S}_2$ or $\mathcal{S}_3$ were linked to more than one $\mathcal{S}_1$ record.
   Therefore, the mapping from targets to source is a **partial function**:
   $$f: (\mathcal{S}_2 \cup \mathcal{S}_3) \to \mathcal{S}_1 \cup \{\bot\}$$
   where $\bot$ indicates an unlinked orphan record (~26% of $\mathcal{S}_2 \cup \mathcal{S}_3$).
3. **Country Invariance:** 
   $$\phi(u) = \phi(v) \implies \text{country}(u) = \text{country}(v)$$
   Empirically verified across 182,932 ground-truth pairs with zero cross-country links.

#### 2.3 Singletons
A subset $\mathcal{S}_{1, \emptyset} \subset \mathcal{S}_1$ represents **singletons** with no matching records in either $\mathcal{S}_2$ or $\mathcal{S}_3$:
$$Y(s_1) = \emptyset \quad \text{for } s_1 \in \mathcal{S}_{1, \emptyset}$$
Empirically, singletons constitute **~5.59%** of the $\mathcal{S}_1$ reference dataset.

---

### 3. Evaluation Metric & Mathematical Optimization Analysis

Submissions are evaluated on the **Macro-averaged $F_{0.5}$ score** across all Source 1 entities in the test set.

#### 3.1 The $F_{0.5}$ Formula
For a single reference entity $s_1 \in \mathcal{S}_1$, let $Y \subseteq (\mathcal{S}_2 \cup \mathcal{S}_3)$ be the ground-truth set of matched IDs, and $\hat{Y} \subseteq (\mathcal{S}_2 \cup \mathcal{S}_3)$ be the predicted set of matched IDs.

- **Case 1: Ground-Truth Singleton ($|Y| = 0$):**
  $$F_{0.5}(s_1) = \begin{cases} 1.0 & \text{if } |\hat{Y}| = 0 \\ 0.0 & \text{if } |\hat{Y}| > 0 \end{cases}$$
  *Implication:* Falsely predicting even a single match for a true singleton causes a catastrophic drop from $1.0 \to 0.0$.

- **Case 2: Ground-Truth Non-Singleton ($|Y| > 0$):**
  Let $TP = |Y \cap \hat{Y}|$, $FP = |\hat{Y} \setminus Y|$, $FN = |Y \setminus \hat{Y}|$.
  If $TP = 0$, then $F_{0.5}(s_1) = 0.0$.
  If $TP > 0$:
  $$\text{Precision} = \frac{TP}{|\hat{Y}|} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{|Y|} = \frac{TP}{TP + FN}$$
  The $F_{\beta}$ score with $\beta = 0.5$ ($\beta^2 = 0.25$) is defined as:
  $$F_{0.5} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

  Substituting Precision and Recall:
  $$F_{0.5} = \frac{1.25 \cdot \frac{TP}{|\hat{Y}|} \cdot \frac{TP}{|Y|}}{0.25 \cdot \frac{TP}{|\hat{Y}|} + \frac{TP}{|Y|}} = \frac{1.25 \cdot TP^2}{0.25 \cdot TP \cdot |Y| + TP \cdot |\hat{Y}|}$$
  Dividing numerator and denominator by $0.25 \cdot TP$:
  $$F_{0.5} = \frac{5 \cdot TP}{|Y| + 4 \cdot |\hat{Y}|}$$

#### 3.2 Key Mathematical Insights from the Metric:
1. **$4\times$ False Positive Penalty:**
   $$\frac{\partial F_{0.5}}{\partial |\hat{Y}|} = -\frac{20 \cdot TP}{(|Y| + 4 |\hat{Y}|)^2}$$
   Adding a single false positive increases the denominator by $4$, drastically suppressing the score.
2. **Threshold Tuning Bias:** The decision threshold $\tau$ must be tuned towards **high precision** ($\ge 0.85$ target precision). 
3. **Singleton Guard:** If the maximum match probability for an entity $s_1$ falls below a dedicated threshold $\tau_{\text{singleton}}$, predicting $\hat{Y} = \emptyset$ yields expected score $1.0 \times P(s_1 \in \mathcal{S}_{1, \emptyset}) \approx 0.056$ instead of risking $0.0$.

---

### 4. Dataset Properties & Empirical Findings

| Split | File | Record Count | Countries | Addr Missing (%) |
|---|---|---|---|---|
| **Train** | `train_source1.tsv` | 2,206,821 | US (60.0%), India (40.0%) | 0.0% |
| **Train** | `train_source2.tsv` | 5,034,616 | US, India | 3.36% |
| **Train** | `train_source3.tsv` | 5,285,603 | US, India | 3.33% |
| **Train** | `train_ground_truth.tsv`| 2,206,821 | US, India | N/A |
| **Test** | `test_source1.tsv` | 1,732,544 | India (46.7%), US (38.3%), France (15.0%) | 0.0% |
| **Test** | `test_source2.tsv` | 4,887,273 | India, US, France | 2.65% |
| **Test** | `test_source3.tsv` | 5,082,316 | India, US, France | 2.68% |

#### Key Data Observations:
1. **Zero Nulls in Source 1:** All reference entities contain valid names and addresses.
2. **Sparse Addresses in Sources 2 & 3:** ~3% of records in $\mathcal{S}_2$ and $\mathcal{S}_3$ have missing (`None`) addresses, requiring robust fallback on business name matching.
3. **Cross-Lingual Transliteration (India):** Hindi (Devanagari) and Tamil scripts appear in $\mathcal{S}_2$ and $\mathcal{S}_3$ while $\mathcal{S}_1$ names are in English. However, street numbers, PIN codes, landmarks, and city names remain in Latin script in the address fields.
4. **Out-of-Distribution (OOD) Country Shift (France):** France accounts for 259,452 records in `test_source1.tsv` (15%), but is completely unobserved in the training set. Pipeline features must be language-invariant, utilizing character n-grams, legal suffix canonicalization, and accent folding (NFKD).
5. **Concatenated Domains / DBA Names:** Domain name strings (e.g. `maurewilliamscolombier.com`) and trade names occur frequently in $\mathcal{S}_3$.

---

### 5. Functional Requirements

1. **Dual Output Generation:**
   - `output/candidate_pairs.tsv`: Tab-separated candidate pool generated by blocking (subset fed to matcher).
   - `output/matching_results.tsv`: Tab-separated final entity resolution predictions.
2. **Formatting & Schema Invariants:**
   - Both files must contain exactly 1,732,544 rows (one per entity in `test_source1.tsv`).
   - TSV formatting (`sep='\t'`), no quoted commas.
   - IDs must only be valid `S2-` and `S3-` identifiers present in the test set.
   - Exact subset guarantee: $\forall s_1, \quad \hat{Y}(s_1) \subseteq \text{Candidates}(s_1)$.
3. **Local Self-Verification:**
   - Must pass `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` with zero errors.

---

### 6. Non-Functional & Operational Requirements

1. **Hardware & Resource Envelope:**
   - Host: 12-core Intel i5-13420H, 15 GB RAM, 98 GB NVMe disk.
   - GPU: RTX 3050 Laptop GPU (6 GB VRAM).
   - Compute Execution Budget: High-efficiency execution via chunked streaming (Polars / PyArrow) and parallel CPU multiprocessing to prevent OOM errors on 26.4 million records.
2. **Fair Play & Integrity Compliance:**
   - Strictly zero external lookups (no external APIs, Google Maps, OpenStreetMap, geocoders, or web queries).
   - Self-contained, reproducible training and inference.
3. **Model License & Size Budget:**
   - Permissive open-source license (MIT / Apache 2.0).
   - Model parameter count $\le 8\text{B}$.
