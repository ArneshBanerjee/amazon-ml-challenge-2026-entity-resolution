# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat the task as "for every Source 2 / Source 3 record, find its one Source 1 entity, or none".
Candidates come from a fine-tuned multilingual bi-encoder searched in both directions and in four
views, then a small LightGBM pruning model cuts them to **{{CPS}} candidates per S1 at {{BREC}} pair
recall**. A two-stage LightGBM stacker combines string features, two cross-encoders and
group-consistency features. Two independent training runs of the whole pipeline are averaged. A
one-to-one assignment and a per-entity set choice that maximizes expected F0.5 give the final lists. Holdout macro F0.5 is **{{C1}}** (tuning part) and **{{C2}}**
(untouched part). For France, which has no labels, address and vocabulary maps are mined from our own
confident test predictions.

---

## 2. Methodology

### 2.1 Problem Analysis

Facts checked on the training data:

- 2,206,821 train S1 entities and 10.3M S2/S3 records. 7,638,365 true pairs. **Every S2/S3 record
  matches at most one S1 entity**, and no true pair crosses country labels. 26% of records match
  nothing (distractors). 5.6% of S1 entities have no match. Each S1 has 0 to 5 S2 matches and 0 to 6 S3
  matches (3.47 on average).
- Test has 5.75 records per S1 (train 4.68). We predict about 3.4 matches per S1 on test for US and
  India, the same as on the holdout, so the extra test records are mostly easy distractors (they get
  near-zero scores).
- Name noise: typos, injected accents, digit-for-letter swaps (`c1ub`, `5arl`), junk prefixes and
  suffixes (`-- `, `<< `, `| www.x.com`), legal forms added, dropped or moved to the front, repeated
  words, domain-style names (`douswilliams.com`, 3% of records), social handles (`@name`), honorifics
  (Sri, M/s), trade-name aliases (`X dba Y`, `t/a`, `formerly`, `doing business as`, 1.5% of S3), and
  records with a completely different trade name at the same address.
- India: 28% of S2 names and 18% of S3 names are in native scripts (Devanagari, Tamil, Malayalam,
  Telugu, Kannada, Odia, Bengali); state names also appear in native script or as codes (TN, UP).
- Address noise: reordered components, `<NULL>` / `null` / `N/A` fillers (2 to 3%), empty addresses
  (2.5 to 3.7% of S2/S3), leading zeros and dropped leading digits (2880 vs 880), ranges (1056-1060),
  `#` / `###`, `Unit` / `PO Box` / landmark additions, street and state abbreviations, city replaced by a
  county or neighbourhood. Postal codes are almost never present.
- Distractors are near copies of a real entity: a nearby house number (1024 vs 1027), one real word
  swapped (First vs Seventh, Private vs Public), a legal form added, or words like "& Sons",
  "& Associates", "Group", "Enterprises" added. In train, a record that adds one of those words to the
  S1 name at the same address is a distractor 98 to 100% of the time, while "Services" and "Center"
  are ordinary noise (about 45% true).
- About half of all S1 names are shared by at least two S1 entities in the same country (for example
  four entities called "Bledsoe Yield" at different addresses).
- France (test only, 15% of test S1): same noise with French patterns (`R.`, `Av`, `Bd`, `Imp`, `Q.`,
  `N°`, SARL / SAS / SASU / EURL / SCI, "& Fils", "& Associés", "Groupe"). Every French S1 address
  ends with a region ("Hauts-de-France"); records drop it in 65% of cases or give the department
  ("Nord", "Gironde") instead. 13% of French S1 entities share an exact address with another one
  (4 to 6% in train).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruning + two-stage stacked classifier + constrained set selection
**Core Innovation:** record-centric matching that uses "each record belongs to at most one entity"
everywhere (reverse kNN, record-side context features, one-to-one assignment), group features that
check whether a record agrees with the other confident records of the same entity, and data-mined
normalization maps, including maps for a country without labels mined from confident predictions.

Validation: train S1 entities are split by country into A (80%, neural models), B (10%, stacker and
pruning model), C1 (8%, calibration and set-selection tuning) and C2 (2%, never used for any choice).
Retrieval for B and C searches the full train S2/S3 pool, distractors included. Negatives used to
train the neural models never include records of B or C entities. We checked that records of entities
seen in training (A) and unseen ones (B, C) are wrongly assigned at the same rate (24 per 10k), so the
holdout is not optimistic because of the neural models.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. Fine-tuned `intfloat/multilingual-e5-small` bi-encoder (MIT). Round 1: in-batch negatives
     (symmetric InfoNCE), batches of 1,024 pairs from one country with unique S1 entities. Round 2:
     continued with mined hard negatives and three views (full record 60%, name only 20%, address only
     20%). Input is the normalized `name | address`.
  2. Exact GPU kNN inside each country string (the country is only a partition key, so France works
     like any other label), forward (S1 to top-k records) and reverse (record to top-k S1). Views:
     round-2 full (k = 20 forward / 3 reverse), name (10 / 2), address (10 / 2), round-1 full (20 / 3).
  3. Exact keys: acronym of the S1 core name = record name (for `GAS`), and domain stem = S1 core name
     without spaces.
  4. A LightGBM pruning model trained on split B with the cosine of every view, both ranks of every
     view, four fast rapidfuzz scores and rank / gap context. Pairs with score >= 0.002 are kept, at
     most 15 per S1. This set is `candidate_pairs.tsv` and is exactly what the matcher scores.
- **Candidate pairs generated:** {{TEST_CANDS}} on test ({{TEST_CPS}} per S1).
- **How we ensured true matches were not lost:** union of views and directions before pruning (C1
  pair recall 99.92% at 58 candidates per S1), then the pruning threshold is chosen on C1 for 99.8%
  pair recall. Recall against candidate count on C1:

| Pruning threshold | Pair recall | Candidates per S1 |
|---|---|---|
| 0.5 | 0.9738 | 3.43 |
| 0.1 | 0.9875 | 3.63 |
| 0.02 | 0.9944 | 4.25 |
| 0.01 | 0.9967 | 4.77 |
| 0.005 | 0.9977 | 5.25 |
| **0.002 (used)** | **0.9982** | **5.8** |
| 0.001 | 0.9984 | 6.26 |
| 0.0001 | 0.9986 | 7.17 |

For comparison, the untrained e5 model gives 96.4% at 4.7 candidates (reverse top-1) and 98.7% at 60.

---

## 4. Matching Model

**Features used:**
- Name features: rapidfuzz ratio, partial ratio, token set / sort and Jaro-Winkler on the normalized
  name, the core name (legal forms and honorifics removed) and the raw lowercase name; alias-name
  ratio; core without spaces vs domain stem (ratio, partial, prefix); legal form equal / missing;
  idf-weighted token overlap; word-swap signal (smallest idf among record tokens with no fuzzy partner
  in the S1 name: a common word suggests a swapped word, a rare one a typo).
- Address features: ratio, token set / sort, partial, partial token set; idf-weighted overlap; number
  features (first number equal, number Jaccard, suffix match for dropped digits, unexplained numbers,
  size of the house-number difference); phone digits equal; empty-address flags.
- Other: bi-encoder cosines of four views and their ranks in both directions, pruning score, source,
  lengths, script flags; rank / gap / margin of the main scores within the S1 and within the record;
  the two cross-encoder scores with the same context features; stage-2 group features (probability of
  the record's best competing S1 and the margin, number of confident records of the S1, similarity of
  the record to the other confident records of the S1 in embedding, name, core name and address, share
  of those records with the same house number, whether the S1's own house number agrees with them).

**Model type:**
- Cross-encoders: `microsoft/mdeberta-v3-base` (MIT, 278M parameters), one on normalized text and one on
  raw text, each trained one epoch on 3M split-A candidate pairs (40% positives, hard negatives from
  the round-1 kNN lists). Removing the raw-text model lowers the holdout from 0.9924 to 0.9917.
- Stacker: LightGBM, 4-fold by S1 on split B; out-of-fold stage-1 predictions feed the stage-2 group
  features. Stage 2 adds about +0.0002.
- Ensemble: the whole pipeline (bi-encoders, cross-encoders, pruning model, stackers) is trained twice
  and the stage-2 logits are averaged (a pair that one run did not keep as a candidate counts as
  probability 1e-4 there); the candidate file is the union of both runs' candidate sets, which is the
  set the ensemble scores. Two runs give 0.99226 and 0.99235 on C1 alone and 0.99254 averaged. The
  gain is larger where single runs disagree: on French test entities two runs differ on 8% of the
  output rows, against 0.8% for US and India.
- All models are MIT or Apache 2.0 and far below 8B parameters.

**Threshold selection method:** one-to-one assignment (each record kept only for its highest-scoring
S1), isotonic calibration fitted on C1, then for every S1 the top-k set (k = 0 allowed) that maximizes
expected F0.5 under the calibrated probabilities. The empty set wins when the probability of "no true
match" beats the best non-empty set, which handles singletons directly. We compared a global
threshold, an approximate and an exact (Poisson-binomial) expected F0.5; the choice is made on C1
(they differ by less than 0.0001).

**Country without labels (France).** After a first full test pass, confident French pairs
(probability >= 0.98 and a margin of 0.9 over the next S1; 792k pairs) are used to mine French
address maps with the same miner as for train: components that are frequent and present on only one
side of most confident pairs are treated as optional (the three regions and four departments), and
spelling aliases are mined (`st nazaire` to `saint nazaire`). A short generic list maps French business
words to English (associés, fils, frères, groupe, développement, entreprises, et, centre), so patterns
learned on English names ("& Sons" added = distractor) apply to French names. The test side is then
normalized again and scored with the unchanged models. Before this step half of the accepted French
pairs had an address token-set similarity below 90; after it 1.8%.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** {{RESULTS}}
- **Common false positives (wrong merges):** distractors that are the S1 name plus a legal form
  (Inc, Corp, LLC) at the same address. True records get an added legal form just as often, and
  sibling records share it no more often for true records than for distractors (4.2% vs 4.0%), so
  these look irreducible. Also names differing by one real word and house numbers differing by a few
  units.
- **Common false negatives (missed matches):** records with an empty address. True pairs with an
  address are recovered 99.8% of the time, empty-address ones 64%. Of the missed empty-address pairs,
  8,664 (C1) have a name shared by another S1 in the same country and only 874 have a unique name.
  The best weak signal we found (the correct S1 has fewer records from the same source) is right 39%
  of the time, far below what F0.5 needs to accept, so predicting nothing is the right choice.

Holdout (macro F0.5):

{{MILESTONES}}

Things we tried that did not help on the holdout: a France-adapted cross-encoder trained on
pseudo-labels (0.99236, same), ambiguity counts (how many S1 share the address or name, 0.99231, same),
dropping the raw-text cross-encoder (0.99171, worse), dropping length features (worse transfer).

Leaderboard lessons: the first full model scored 0.98798 on the public leaderboard against 0.992 on
our US / India holdout. Accepted US and India test pairs have the same profile as holdout pairs (same
word-swap, number-change, legal-form and empty-address rates and the same mean probability), so the gap
is most likely France. Raising French acceptance to the US / India match rate lowered the score
(0.986), so the French errors are not missed matches of the uncertain kind; this led to the
French address and vocabulary maps described above.

Cross-country transfer check (stacker trained on one country, scored on the other, C1): with the final
feature set US to India 0.986 and India to US 0.982, against 0.933 and 0.958 with the first baseline
features. The bi-encoder features transfer well.

---

## 6. Conclusion

A fine-tuned bi-encoder searched in both directions and several views, followed by a learned pruning
model, gives 99.8% recall with under 6 candidates per entity. Cross-encoders, group-consistency
features and a one-to-one, expected-F0.5 set choice bring the holdout above 0.992. The remaining
holdout errors come from the data itself (identical names without an address, distractors identical
to noisy copies). The hardest part was the unseen country: normalization and vocabulary that were
never needed for US and India decide how well the models transfer.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds all code in `src/`, a `README.md` with setup and run
commands, and a pinned `requirements.txt`. One command regenerates both output files from the raw data:

```bash
PY=$PWD/.venv/bin/python bash src/run_all.sh /path/to/student_resource/dataset /path/to/output
```

Stages: `load.py` (TSV to parquet), `splits.py`, `normalize.py` (rules plus maps mined from split-A
pairs), `train_biencoder.py` / `encode.py` / `knn.py` (retrieval), `block.py` / `prune.py` (candidates),
`features.py` / `featurize.py`, `cross.py` (cross-encoders), `stack.py` / `group.py` (stacker),
`pseudo.py` (maps for countries without labels), `ensemble.py` (average of runs), `select_sets.py`
(one-to-one, calibration, set selection, output files), `evaluate.py` (metric). `run_all.sh` is one
complete run; `run_ensemble.sh` runs it twice (seeds 0 and 1) and averages them. The README has a
table of all scripts.

### B. Additional Results

Normalization details: NFKC, anyascii transliteration (ISC license), a mined native-to-Latin word map
(555 entries, e.g. `limirrd` to `limited`), mined state and city aliases (`tmilnatu` / `tn` to tamil
nadu), mined street abbreviations and typo fixes (`st` to street, `aveune` to avenue), a small generic
list of English and French street abbreviations and French business words, and legal forms mapped to
one token and kept in a separate field. All mined maps are keyed by the exact country string.

Blocking recall of the fine-tuned round-1 bi-encoder alone on C (forward top-k, reverse top-k):
k = 10: 99.44%, k = 20: 99.80%, k = 30: 99.88%, reverse top-1: 97.9% at 4.7 candidates.

Models and licenses: `intfloat/multilingual-e5-small` (MIT, 118M), `microsoft/mdeberta-v3-base` (MIT,
278M), LightGBM (MIT). Libraries: polars (MIT), pyarrow (Apache 2.0), numpy (BSD), rapidfuzz (MIT),
torch (BSD-style), transformers (Apache 2.0), anyascii (ISC), scikit-learn (BSD).
