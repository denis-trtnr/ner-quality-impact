from typing import Dict, Tuple
from datasets import load_dataset, ClassLabel

def load_conll2003():
    return load_dataset("conll2003")

def build_label_maps(features) -> Tuple[Dict[int, str], Dict[str, int]]:
    ner_feature: ClassLabel = features["ner_tags"].feature
    id2label = {i: ner_feature.names[i] for i in range(ner_feature.num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def tokenize_and_align(examples, tokenizer, label_all_tokens=False, max_length=None):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    labels = []
    for i, word_ids in enumerate(tokenized.word_ids(batch_index=None)):
        word_to_label = examples["ner_tags"][i]
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(word_to_label[word_id])
            else:
                label_ids.append(word_to_label[word_id] if label_all_tokens else -100)
            prev_word_id = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized