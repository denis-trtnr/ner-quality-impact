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

# wandb

wandb sweep sweep_config.yaml

bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh wandb agent <SWEEP_ID>

bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh wandb agent z8w3kb4e

bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh wandb agent denistrautner-dhbw-duale-hochschule-baden-w-rttemberg/ner-quality-impact/<SWEEP_ID>

bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh wandb agent denistrautner-dhbw-duale-hochschule-baden-w-rttemberg/ner-quality-impact/2u9hbr6g

wandb sweep --stop denistrautner-dhbw-duale-hochschule-baden-w-rttemberg/ner-quality-impact/2u9hbr6g