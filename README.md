# ner-quality-impact


conda create -n robust-ner python=3.10 -y

conda activate robust-ner

pip install -e .

python -m src.train --model bert-base-cased --profile src/profiles/typo-basic.yaml

python -m src.train \
    --model bert-base-cased \
    --profile src/profiles/typo-basic.yaml \
    --epochs 1 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_length 256 \
    --seed 42 \
    --out outputs/typo-basic
