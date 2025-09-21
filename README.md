# ner-quality-impact
conda info --envs

conda create -n robust-ner python=3.10 -y

conda activate robust-ner

pip install -e .


# Manually running code
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

# Calling script
bash scripts/run_profile_pegasus.sh bert-base-cased src/profiles/typo-basic.yaml
