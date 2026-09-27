# Business Entity Resolution (ML Challenge 2026)

For every Source 1 business, find the matching Source 2 and Source 3 records.
The pipeline goes from the raw TSV files to `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, using only the provided train and test files.

## Requirements

- Linux, Python 3.12, one NVIDIA GPU with at least 40 GB memory (tested on an H100 80 GB), about
  100 GB RAM, about 150 GB free disk for intermediate files.
- [uv](https://docs.astral.sh/uv/) for the Python environment.
- Internet access once, to download two pretrained models from Hugging Face:
  `intfloat/multilingual-e5-small` (MIT) and `microsoft/mdeberta-v3-base` (MIT).
  No other external data is used.

## Setup

```bash
cd business_entity_resolution
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

## Run

The submitted files come from two independent runs of the pipeline: for countries present in train the
stage-2 scores of both runs are averaged; for countries without labels (France) the rows of run 1 are
kept (see `splice.py` and the documentation for why):

```bash
cd business_entity_resolution
PY=$PWD/.venv/bin/python bash src/run_ensemble.sh /path/to/student_resource/dataset /path/to/output
```

A single run (about half the time, holdout F0.5 about 0.0002 lower) is:

```bash
PY=$PWD/.venv/bin/python bash src/run_all.sh /path/to/student_resource/dataset /path/to/output
```

The dataset folder must contain `train/` and `test/` with the original TSV files. The two result
files are written to the output folder (default `business_entity_resolution/output`).
Intermediate files go to `business_entity_resolution/artifacts/`. Every stage skips work whose
output already exists, so after an interruption the same command continues where it stopped.
One run takes about 8 to 10 hours on one H100 (most of it is transformer training and inference).
LightGBM uses all cores but two by default; set `BER_THREADS` to change that. The folder
`artifacts/output_before_unlabeled_step/` also gets the two files as they are before the step for
countries without labels.

To check the files with the official validator:

```bash
cd /path/to/student_resource
python3 utils/validate_submission.py --matching /path/to/output/matching_results.tsv \
    --candidate /path/to/output/candidate_pairs.tsv --test-dir dataset/test
```

## Stages (all in `src/`)

| Script | What it does |
|---|---|
| `load.py` | Reads the TSVs with every column as a string, checks row counts, stores parquet. |
| `splits.py` | Splits train S1 entities by country into A (80%, neural models), B (10%, stacker), C1 (8%, tuning) and C2 (2%, untouched check). |
| `normalize.py` | Transliteration (anyascii), junk removal, legal forms, alias names (dba, t/a, formerly), domains, address cleanup. Mines alias maps (native script words, state and street abbreviations) from split-A true pairs. |
| `train_biencoder.py`, `encode.py`, `knn.py` | Fine-tunes `multilingual-e5-small` as a bi-encoder (two rounds, the second with mined hard negatives and name-only / address-only views), encodes all records, exact GPU kNN inside each country string in both directions. |
| `block.py`, `prune.py` | Candidate union (4 kNN views + acronym and domain keys), then a LightGBM pruning model. Its output is `candidate_pairs.tsv`. |
| `features.py`, `featurize.py` | String features (rapidfuzz), idf overlaps, number and word-swap signals, rank and gap context. |
| `cross.py` | Cross-encoders (`mdeberta-v3-base`) on normalized and on raw text, trained on split-A candidate pairs. |
| `stack.py`, `group.py` | Two-stage LightGBM stacker trained on split B (4-fold, out-of-fold predictions). Stage 2 adds group features: agreement of a record with the other confident records of the same S1. |
| `pseudo.py` | For test countries without labels (France): takes confident test pairs from the first pass and mines address maps (optional region and department components, spelling aliases) with the same miner used on train. The test side is then normalized again and scored with the unchanged models. |
| `ensemble.py` | Averages the stage-2 logits of several runs (a pair missing from a run counts as 1e-4), candidate set = union of the runs' candidate sets. |
| `splice.py` | Final file: ensemble rows for countries present in train, single-run rows for countries without labels; candidate file = union of both candidate files. |
| `select_sets.py` | One-to-one assignment (each record goes to at most one S1), isotonic calibration on C1, per-S1 set choice that maximizes expected F0.5, writes both output files. |
| `evaluate.py` | Exact macro F0.5 (singletons included), blocking recall and candidates per S1. |
| `analyze.py`, `blockstats.py`, `transfer.py` | Error analysis and checks used during development (not needed for the outputs). |

See `RESULTS.md` for the leaderboard history, which submission scored best, and open problems.

Settings such as the data and artifact folders can be changed with the environment variables
`BER_DATA`, `BER_ART` and `BER_ROOT` (see `src/common.py`).
