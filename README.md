# NER Quality Impact  

This project investigates how **data quality** influences the performance of different **Named Entity Recognition (NER)** models. It systematically introduces **noise** into datasets (train, validation, and test sets) at varying **noise rates** and **stages** to measure robustness and generalization of various NER architectures.

> 📚 **This work was conducted as part of a study in collaboration with the [DFKI Speech & Language Technology Lab](https://www.dfki.de/en/web/research/research-departments/speech-and-language-technology)**

## 🧩 Repository Structure

```bash
ner-quality-impact/
│
├── docs/                     # Research docs
├── notebooks/                # Example notebooks
├── scripts/                  # Execution scripts
│   ├── run_profile.sh
│   └── run_profile_pegasus.sh
├── src/                      # Core source code
│   ├── noise/                # Noise generation modules
│   │  ├── utils/             # Helper utilities for noise generation
│   │  ├── label_noise.py     # Injects noise into labels/entities
│   │  ├── orthographic.py    # Orthographic (character-level) noise
│   │  ├── registry.py        # Registry for available noise types
│   │  ├── semantic.py        # Semantic-level noise (word meaning)
│   │  └── syntactic.py       # Syntactic noise (structure-based)
│   ├── profiles/             # Experiment configurations
│   ├── data_preprocessing.py # Data loading and preprocessing
│   ├── metrics.py            # Evaluation and scoring metrics
│   └── train.py              # Training loop and orchestration
├── requirements.txt          # Dependencies
├── sweep_config_*.yaml       # W&B sweep configurations
└── README.md                 # You're looking at it :-)

```

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/denis-trtnr/ner-quality-impact.git
```
```bash
cd ner-quality-impact
```
```bash
pip install -r requirements.txt
```

## 🚀 Running Experiments

You can run experiments in multiple ways depending on your setup.

### 1️⃣ Manual Execution
```bash
python -m src.train \
    --model bert-base-cased \
    --profile src/profiles/<PROFILE> \
    --epochs 5 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_length 256 \
    --seed 42
```
---


### 2️⃣ Using Provided Shell Scripts
💻 Local Execution
```bash
bash scripts/run_profile.sh bert-base-cased src/profiles/orthographic/orthographic_p0.1_test_all.yaml

```
🦄 [Pegasus](https://pegasus.dfki.de/) Cluster 
```bash
bash scripts/run_profile_pegasus.sh bert-base-cased src/profiles/orthographic/orthographic_p0.1_test_all.yaml

```
---

### 3️⃣ Running W&B Sweeps
This project uses **grid search sweeps** with **Weights & Biases (W&B)** to automate structured experiments.  
Each agent executes **one training run after another**, iterating through all defined configurations in sequence.

Example sweep config files:
- `sweep_config_baseline.yaml`
- `sweep_config_test.yaml`
- `sweep_config_train_validation_test.yaml`
- `sweep_config_train_validation.yaml`

Run a sweep:

```bash
wandb sweep sweep_config_baseline.yaml
wandb agent <YOUR_SWEEP_ID>
```

Example for using on cluster (using [Pegasus Bridle Wrapper](https://github.com/DFKI-NLP/pegasus-bridle)):
```bash
bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh wandb agent <YOUR_SWEEP_ID>
```


