import random
from typing import List, Tuple

# Simple punctuation perturbations
def punct_insert(tokens: List[str], labels: List[int], prob: float = 0.05, o_label: int = None) -> List[str]:
    """
    Insert random punctuation after tokens with probability `prob`.
    Inserted punctuation always gets label 'O' (id = o_label).
    """
    puncts = [",", ".", ";", ":", "!", "?"]
    out_tokens, out_labels = [], []
    for t, l in zip(tokens, labels):
        out_tokens.append(t)
        out_labels.append(l)
        if random.random() < prob:
            out_tokens.append(random.choice(puncts))
            out_labels.append(o_label)  # always 'O' for inserted punct
    return out_tokens, out_labels

def punct_delete(tokens: List[str], labels: List[int], prob: float = 0.1) -> Tuple[List[str], List[int]]:
    """
    Delete punctuation tokens with probability `prob`.
    Remove both token and its label (usually 'O').
    """
    out_tokens, out_labels = [], []
    for t, l in zip(tokens, labels):
        if t in ",.;:!?" and random.random() < prob:
            continue  # drop token and its label
        out_tokens.append(t)
        out_labels.append(l)
    return out_tokens, out_labels

# Whitespace merge/split simulated via token joins/splits (lightweight)
def whitespace_merge(tokens: List[str], labels: List[int], prob: float = 0.05) -> Tuple[List[str], List[int]]:
    """
    Merge two adjacent tokens into one with probability `prob`.
    Keep the label of the first token in the merge, drop the second.
    """
    out_tokens, out_labels = [], []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        if i < len(tokens) - 1 and random.random() < prob:
            # merge tokens i and i+1
            out_tokens.append(tokens[i] + tokens[i+1])
            out_labels.append(labels[i])  # keep first label
            skip = True
        else:
            out_tokens.append(tokens[i])
            out_labels.append(labels[i])
    return out_tokens, out_labels