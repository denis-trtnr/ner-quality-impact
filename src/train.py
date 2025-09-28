import argparse
import os
import random
import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed
)
from datasets import DatasetDict

from .data_preprocessing import load_conll2003, build_label_maps, tokenize_and_align, tokenize_and_align_chars
from .metrics import compute_metrics_builder
from .noise import TOKEN_NOISE, LABEL_NOISE

def seed_all(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_profile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def build_mappers(profile, id2label, label2id):
    token_steps = profile.get("token_noise", [])
    label_steps = profile.get("label_noise", [])

    def token_mapper(example):
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        for step in token_steps:
            name = step["name"]
            fn = TOKEN_NOISE[name]
            params = step.get("params", {})
            # adapt signatures per function
            if name == "typo_tokens":
                tokens = fn(tokens, ner_tags, id2label, **params)
            elif name in ("synonym_substitute",):
                tokens = fn(tokens, ner_tags, id2label, **params)
            elif name in ("punct_delete", "whitespace_merge"):
                tokens, ner_tags = fn(tokens, ner_tags, **params)
            elif name in ("punct_insert"):
                tokens, ner_tags = fn(tokens, ner_tags, o_label=label2id["O"], **params)
            elif name in ("syntactic_noise"):
                tokens, ner_tags = fn(tokens, ner_tags, o_label=label2id["O"], id2label=id2label, label2id=label2id, **params)
            else:
                # word-level ops not used directly; keep for extensibility
                pass
        return {"tokens": tokens, "ner_tags": ner_tags}

    def label_mapper(example):
        ner_tags = example["ner_tags"]

        for step in label_steps:
            name = step["name"]
            fn = LABEL_NOISE[name]
            params = step.get("params", {})
            ner_tags = fn(ner_tags, id2label, label2id, **params)
        return {"ner_tags": ner_tags}

    return token_mapper, label_mapper

def apply_profile(ds: DatasetDict, profile, id2label, label2id):
    scope = profile.get("scope", {})
    token_scopes = scope.get("token_noise", []) # e.g., ["test"] or ["train","test"]
    label_scopes = scope.get("label_noise", [])

    token_mapper, label_mapper = build_mappers(profile, id2label, label2id)

    if token_scopes:
        if "train" in token_scopes and profile.get("token_noise"):
            ds["train"] = ds["train"].map(token_mapper)
        if "validation" in token_scopes and profile.get("token_noise"):
            ds["validation"] = ds["validation"].map(token_mapper)
        if "test" in token_scopes and profile.get("token_noise"):
            ds["test"] = ds["test"].map(token_mapper)
    if label_scopes:
        if "train" in label_scopes and profile.get("label_noise"):
            ds["train"] = ds["train"].map(label_mapper)
        if "validation" in label_scopes and profile.get("label_noise"):
            ds["validation"] = ds["validation"].map(label_mapper)
        if "test" in label_scopes and profile.get("label_noise"):
            ds["test"] = ds["test"].map(label_mapper)
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--profile", required=True, help="YAML file with noise steps & scopes")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--dense_train", action="store_true", help="Use dense labels during training")
    args = ap.parse_args()

    seed_all(args.seed)

    ds = load_conll2003()
    id2label, label2id = build_label_maps(ds["train"].features)

    profile = load_profile(args.profile)
    ds = apply_profile(ds, profile, id2label, label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Special case for RoBERTa-like models
    if "roberta" in args.model.lower() or "deberta" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    # Different tokenization functions for different models and train/eval modes
    def tok_map_train_normal(b): 
        return tokenize_and_align(b, tokenizer, label_all_tokens=args.dense_train, max_length=args.max_length)
    def tok_map_eval_normal(b): 
        return tokenize_and_align(b, tokenizer, label_all_tokens=False, max_length=args.max_length)
    def tok_map_train_char_level(b):
        return tokenize_and_align_chars(b, tokenizer, id2label, label2id, max_length=args.max_length,
                                     eval_mode=not args.dense_train)  # False => dense; True => first-char
    def tok_map_eval_char_level(b):
        return tokenize_and_align_chars(b, tokenizer, id2label, label2id, max_length=args.max_length,
                                     eval_mode=True)  # always first-char for fair eval

    if "canine" in args.model.lower():
        tokenized = DatasetDict({
            "train": ds["train"].map(tok_map_train_char_level, batched=True),
            "validation": ds["validation"].map(tok_map_eval_char_level, batched=True),
            "test": ds["test"].map(tok_map_eval_char_level, batched=True),
        })
    else:
        tokenized = DatasetDict({
            "train": ds["train"].map(tok_map_train_normal, batched=True),
            "validation": ds["validation"].map(tok_map_eval_normal, batched=True),
            "test": ds["test"].map(tok_map_eval_normal, batched=True),
        })

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    os.makedirs(args.out, exist_ok=True)

    # Create metadata for W&B
    profile_name = os.path.basename(args.profile).replace(".yaml", "")
    run_name = f"{args.model}-{profile_name}-seed{args.seed}"

    training_args = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=args.seed,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["wandb"],
        run_name=run_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"])
    print("===== TEST METRICS =====")
    for k, v in test_metrics.items():
        if k.startswith("eval_"):
            print(f"{k.replace('eval_', '')}: {v:.4f}")

if __name__ == "__main__":
    main()