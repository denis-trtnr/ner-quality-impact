#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-bert-base-cased}
PROFILE=${2:-robust_ner/profiles/typo-basic.yaml}
python -m robust_ner.train --model "$MODEL" --profile "$PROFILE" --epochs 5 --batch_size 16 --lr 3e-5 --max_length 256 --seed 42 --out outputs/$(basename "$PROFILE" .yaml)