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
    """
    Tokenize word-level inputs and align labels for subword models (BERT, RoBERTa, etc.).

    Words may split into subwords:
      - first subword gets the word label
      - others get -100 (ignored) unless label_all_tokens=True

    Returns tokenized batch with subword-aligned "labels".
    """
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


def tokenize_and_align_chars(
    batch,
    tokenizer,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    max_length: int = 1024,
    eval_mode: bool = False,
):
    """
    CANINE char-level alignment for word-labeled NER.

    Train (eval_mode=False):
      - Expand word labels across all characters.
      - Insert 'O' between words (spaces).
      - Padding -> -100.

    Eval/Test (eval_mode=True) [FAIR WORD-LEVEL EVAL]:
      - Only the FIRST character of each word gets the word label.
      - All remaining characters of that word -> -100 (ignored).
      - Spaces between words -> -100 (ignored).
      - Padding -> -100.

      --> Ground truth + seqeval are word based → give exactly one tag per word
      --> Avoids tokenizer/length bias (longer words don’t get extra “votes”).
      --> Mirrors the standard “first-subtoken” eval used for BERT/RoBERTa.
    """
    texts = [" ".join(tokens) for tokens in batch["tokens"]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    new_labels = []
    for words, word_labels in zip(batch["tokens"], batch["ner_tags"]):
        char_labels = []

        for wi, (word, lab_id) in enumerate(zip(words, word_labels)):
            lab_str = id2label[lab_id]

            if eval_mode:
                # EVAL/TEST: one label per word, at the first char only
                if len(word) > 0:
                    char_labels.append(lab_id)                 # first char carries the word tag
                    char_labels.extend([-100] * (len(word)-1)) # rest ignored
                # ignore space after word
                if wi < len(words) - 1:
                    char_labels.append(-100)
            else:
                # TRAIN: dense char supervision
                if lab_str.startswith("B-"):
                    etype = lab_str[2:]
                    char_labels.append(lab_id)  # first char = B-type
                    char_labels.extend([label2id[f"I-{etype}"]] * (len(word) - 1))
                else:
                    # O or I-* repeated for all chars
                    char_labels.extend([lab_id] * len(word))
                # space as O
                if wi < len(words) - 1:
                    char_labels.append(label2id["O"])

        # pad/truncate
        char_labels = char_labels[:max_length]
        char_labels.extend([-100] * (max_length - len(char_labels)))
        new_labels.append(char_labels)

    tokenized["labels"] = new_labels
    return tokenized
