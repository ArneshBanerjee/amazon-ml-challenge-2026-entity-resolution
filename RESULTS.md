# Results and notes

Read this before changing anything. It records what was submitted, which file scored best, and
what was tried.

## Leaderboard history (public leaderboard, macro F0.5)

| Upload | What it is | Score |
|---|---|---|
| 1. M3 | single run, v2 candidates, 2 cross-encoders, two-stage stacker, expected-F0.5 selection | 0.987978 |
| 2. M5 | M3 with a logit offset for France (+3.19) so French matches per S1 equal the US / India rate | 0.986 (worse) |
| 3. **F1 (best)** | **ensemble of two runs for US and India, M3 rows for France** (`run_ensemble.sh`, `splice.py`) | **best of the three** (exact value not recorded here) |

For reference on the day: top team 0.991483, 10th place 0.990282.

**Best file: F1.** It is the `output/matching_results.tsv` in the final submission zip. Compared with
M3 it changes 6,009 US / India rows and no French rows.

## Holdout scores (train split C, never used for training)

| Model | C1 (tuning) | C2 (untouched) |
|---|---|---|
| Baseline: pretrained e5 kNN, rapidfuzz features, LightGBM | 0.97784 | 0.97823 |
| Fine-tuned bi-encoder blocking, pruning, 2 cross-encoders, stage-1 stacker | 0.99211 | 0.99198 |
| + stage-2 group features, one-to-one, expected F0.5 (M3) | 0.99235 | 0.99207 |
| Same pipeline re-run from scratch with this code | 0.99226 | 0.99207 |
| Average of the two runs (used for US / India in F1) | 0.99254 | 0.99219 |

Blocking: 99.82% pair recall at 5.8 candidates per S1 (C1).

## The open problem: the gap between the holdout and the leaderboard

The holdout (US / India) says about 0.992, the leaderboard says about 0.988. Checks done:

- Test US / India accepted pairs have the same profile as holdout accepted pairs (word swaps, house
  number changes, legal-form changes, empty addresses, mean probability), and the same number of
  matches per S1 (3.39). So US / India on test most likely score like the holdout.
- That puts France (15% of test S1, no labels) at roughly 0.96.
- Accepting more French pairs (M5) made it worse: the extra pairs are about half correct.
- Using M5 to recalibrate French probabilities, the best possible French threshold policy gains at
  most about +0.0003 overall. So the French loss is mostly pairs the model is confidently wrong about,
  not threshold choice.
- France facts: every French S1 address ends with a long region that records often drop or replace by
  a department (fixed by the optional `UNLABELED_STEP=1` in `run_all.sh`; it changed ~11k French rows
  and was not submitted, effect unknown). 13% of French S1 entities share an exact address with another
  one (4 to 6% in train). French distractor words ("& Fils", "& Associés", "Groupe") differ from the
  English ones the model learned ("& Sons", "& Associates", "Group"); `normalize.py` maps them.
- Two independent runs disagree on 8% of French output rows (0.8% for US / India).

Ideas not tried (no labels or uploads left to test them): a model trained on US only and checked on
India with French-like noise injected; features for the number of S1 entities sharing an address
together with the record's name; a dedicated French distractor detector.

## Things that did not help on the holdout

France-adapted cross-encoder from pseudo-labels (0.99236), ambiguity counts such as how many S1 share an
address or a name (0.99231), dropping the raw-text cross-encoder (0.99171, worse), dropping length
features (worse US to India transfer), adding stacker variants to the two-run ensemble (0.99244 to
0.99250), exact vs approximate expected-F0.5 selection (same).

## Practical notes

- One run of `run_all.sh` takes 8 to 10 hours on one H100; `run_ensemble.sh` runs it twice.
- LightGBM becomes about 40 times slower when another job uses some CPU cores and LightGBM uses all of
  them. The default is now all cores but two (`BER_THREADS`).
- transformers 5 loads mdeberta-v3 in fp16 by default; training it that way gives NaN. The code loads
  it in fp32 and trains with bf16 autocast.
- Multiprocess DataLoader workers deadlocked after polars had started its threads; tokenization runs in
  a thread pool instead (`common.prefetch`).
- Keep other GPU users in mind: cosine computations keep embeddings in CPU memory and move chunks to
  the GPU, which avoids out-of-memory errors when the GPU is shared.
