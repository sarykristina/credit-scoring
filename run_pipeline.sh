#!/usr/bin/env bash
set -euo pipefail

mkdir -p work outputs

python src/credit_scoring_pipeline.py --skip-baseline
python src/credit_scoring_improve.py --skip-extra-trees
python src/build_sequence_features.py
python src/build_distribution_features.py

python src/credit_scoring_lgbm.py \
  --basic-features 260 \
  --extra-features 240 \
  --seq-features 180 \
  --dist-features 220 \
  --max-train 720000 \
  --val-size 240000 \
  --seed 807 \
  --model-set rankish \
  --pos-scale-mult 0.45
