#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-bert-base-cased}
PROFILE=${2:-src/profiles/typo-basic.yaml}
MAX_LENGTH=${3:-256}

bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh \
  python -m src.train \
    --model "$MODEL" \
    --profile "$PROFILE" \
    --epochs 5 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_length "$MAX_LENGTH" \
    --seed 42 \
    --out "./outputs"