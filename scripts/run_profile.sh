#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-bert-base-cased}
PROFILE=${2:-src/profiles/typo-basic.yaml}
MAX_LENGTH=${3:-256}
python -m src.train --model "$MODEL" --profile "$PROFILE" --epochs 1 --batch_size 16 --lr 3e-5 --max_length "$MAX_LENGTH" --seed 42 --out outputs/$(basename "$PROFILE" .yaml)