#!/bin/bash
# Final submission: two independent runs of the full pipeline, then the average of their stage-2 scores.
#
# Usage: bash src/run_ensemble.sh /path/to/student_resource/dataset /path/to/output
# Run 1 uses artifacts/ (seed 0), run 2 uses artifacts_run2/ (seed 1). Each run is resumable on its own.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DATA="${1:?give the dataset folder (with train/ and test/)}"
OUT="${2:-$HERE/../output}"
PY="${PY:-python}"
ROOT="$(cd .. && pwd)"

BER_ART="$ROOT/artifacts" bash run_all.sh "$DATA" "$ROOT/artifacts/output_single_run"
BER_SEED=1 BER_ART="$ROOT/artifacts_run2" bash run_all.sh "$DATA" "$ROOT/artifacts_run2/output_single_run"

export BER_DATA="$DATA" BER_ART="$ROOT/artifacts"
$PY ensemble.py --runs "$ROOT/artifacts,$ROOT/artifacts_run2" --pred s2_v2 --cand v2 --out ens
$PY select_sets.py --pred ens --cand ens --out-dir "$ROOT/artifacts/output_ensemble"
# countries in train: ensemble rows; countries without labels (France): pairs both runs accept (splice.py)
$PY splice.py --labelled "$ROOT/artifacts/output_ensemble" --unlabelled "$ROOT/artifacts/output_single_run" \
    --unlabelled2 "$ROOT/artifacts_run2/output_single_run" --mode intersect --out-dir "$OUT"
echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
