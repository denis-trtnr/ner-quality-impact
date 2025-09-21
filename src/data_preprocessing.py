from typing import Dict, Tuple
from datasets import load_dataset, ClassLabel

def load_conll2003():
    return load_dataset("conll2003", trust_remote_code=True)

def build_label_maps(features) -> Tuple[Dict[int, str], Dict[str, int]]:
    ner_feature: ClassLabel = features["ner_tags"].feature
    id2label = {i: ner_feature.names[i] for i in range(ner_feature.num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def tokenize_and_align(batch, tokenizer, label_all_tokens=False, max_length=256):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    new_labels = []
    for i, labels in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(labels[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        new_labels.append(label_ids)

    tokenized["labels"] = new_labels
    return tokenized