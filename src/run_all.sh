#!/bin/bash
# Full pipeline: raw TSVs -> output/matching_results.tsv and output/candidate_pairs.tsv
#
# Usage: bash src/run_all.sh /path/to/student_resource/dataset [/path/to/output]
# Every stage skips work whose output already exists, so the script can be re-run after an
# interruption and it continues where it stopped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export BER_DATA="${1:?give the dataset folder (with train/ and test/)}"
OUT="${2:-$HERE/../output}"
PY="${PY:-python}"
export BER_ART="${BER_ART:-$HERE/../artifacts}"
A="$BER_ART"
mkdir -p "$A" ../logs

# 1. data, splits, normalization (with alias maps mined from split-A train pairs)
$PY load.py
$PY splits.py
$PY normalize.py

# 2. bi-encoder round 1 (in-batch negatives) and its kNN lists
$PY train_biencoder.py --out bi_e5s_r1 --bs 1024 --epochs 1
$PY encode.py --model $A/models/bi_e5s_r1 --tag r1 --prefix 'query: ' --max-len 64
$PY knn.py --tag r1 --kf 30 --kr 10

# 3. bi-encoder round 2 (hard negatives from round 1, combined / name / address views)
$PY train_biencoder.py --out bi_e5s_r2 --init $A/models/bi_e5s_r1 --hardneg r1 \
    --views combined:0.6,name:0.2,addr:0.2 --bs 1024 --epochs 1 --lr 3e-5
M=$A/models/bi_e5s_r2
$PY encode.py --model $M --tag r2c --field combined --prefix 'query: ' --max-len 64
$PY encode.py --model $M --tag r2n --field name --prefix 'query: ' --max-len 48
$PY encode.py --model $M --tag r2a --field addr --prefix 'query: ' --max-len 64
$PY knn.py --tag r2c --kf 30 --kr 10
$PY knn.py --tag r2n --kf 20 --kr 5
$PY knn.py --tag r2a --kf 20 --kr 5

# 4. cross-encoders, trained on split-A candidate pairs from the round-1 kNN lists
$PY block.py --knn r1 --kf 5 --kr 2 --out r1k --splits train
$PY cross.py train --cand r1k --out ce_mdeb1 --n-pairs 3000000 --bs 128 --lr 3e-5
$PY cross.py train --cand r1k --out ce_mdeb_raw --text raw --seed 1 --n-pairs 3000000 --bs 128 --lr 3e-5

# 5. candidate union + pruning model -> final candidate set v2
$PY prune.py --out v2

# 6. pair features and cross-encoder scores on the candidate set
$PY featurize.py --cand v2
$PY cross.py infer --cand v2 --out ce_mdeb1
$PY cross.py infer --cand v2 --out ce_mdeb_raw

# 7. two-stage stacker (stage 2 adds group features built from stage-1 predictions)
$PY stack.py --feat v2 --extra ce_mdeb1,ce_mdeb_raw --out s1_v2
$PY group.py --pred s1_v2 --out g_v2
$PY stack.py --feat v2 --extra ce_mdeb1,ce_mdeb_raw --gfeat g_v2 --out s2_v2
# checkpoint: the same two files before the step for unlabelled countries (kept with the artifacts)
$PY select_sets.py --pred s2_v2 --cand v2 --out-dir $A/output_before_unlabeled_step

# 8. countries without training labels (France): mine address maps from confident test pairs
#    (optional region / department components, spelling aliases), re-normalize test, and redo the
#    test side with the saved models (train side and all models stay as they are)
#    Optional (UNLABELED_STEP=1). Tested; not used for the submitted files.
if [ "${UNLABELED_STEP:-0}" = "1" ]; then
    $PY pseudo.py --pred s2_v2
    $PY normalize.py --test-only
    M=$A/models/bi_e5s_r2
    $PY encode.py --model $A/models/bi_e5s_r1 --tag r1 --prefix 'query: ' --max-len 64 --splits test --force
    $PY encode.py --model $M --tag r2c --field combined --prefix 'query: ' --max-len 64 --splits test --force
    $PY encode.py --model $M --tag r2n --field name --prefix 'query: ' --max-len 48 --splits test --force
    $PY encode.py --model $M --tag r2a --field addr --prefix 'query: ' --max-len 64 --splits test --force
    $PY knn.py --tag r1 --kf 30 --kr 10 --splits test --force
    $PY knn.py --tag r2c --kf 30 --kr 10 --splits test --force
    $PY knn.py --tag r2n --kf 20 --kr 5 --splits test --force
    $PY knn.py --tag r2a --kf 20 --kr 5 --splits test --force
    $PY prune.py --out v2 --test-only --force
    $PY featurize.py --cand v2 --splits test --force
    $PY cross.py infer --cand v2 --out ce_mdeb1 --splits test --force
    $PY cross.py infer --cand v2 --out ce_mdeb_raw --splits test --force
    $PY stack.py --feat v2 --extra ce_mdeb1,ce_mdeb_raw --out s1_v2 --test-only
    $PY group.py --pred s1_v2 --out g_v2 --splits test --force
    $PY stack.py --feat v2 --extra ce_mdeb1,ce_mdeb_raw --gfeat g_v2 --out s2_v2 --test-only
fi

# 9. one-to-one assignment, calibration and set selection tuned on C1, then write both files
$PY select_sets.py --pred s2_v2 --cand v2 --out-dir "$OUT"
echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
