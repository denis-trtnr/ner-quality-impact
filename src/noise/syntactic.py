import random
from typing import List, Tuple

# Simple punctuation perturbations
def punct_insert(tokens: List[str], labels: List[int], o_label: int = None, p: float = 0.05) -> List[str]:
    """
    Insert random punctuation after tokens with probability `prob`.
    Inserted punctuation always gets label 'O' (id = o_label).
    """
    puncts = [",", ".", ";", ":", "!", "?"]
    out_tokens, out_labels = [], []
    for t, l in zip(tokens, labels):
        out_tokens.append(t)
        out_labels.append(l)
        if random.random() < p:
            out_tokens.append(random.choice(puncts))
            out_labels.append(o_label)  # always 'O' for inserted punct
    return out_tokens, out_labels

def punct_delete(tokens: List[str], labels: List[int], p: float = 0.1) -> Tuple[List[str], List[int]]:
    """
    Delete punctuation tokens with probability `prob`.
    Remove both token and its label (usually 'O').
    """
    out_tokens, out_labels = [], []
    for t, l in zip(tokens, labels):
        if t in ",.;:!?" and random.random() < p:
            continue  # drop token and its label
        out_tokens.append(t)
        out_labels.append(l)
    return out_tokens, out_labels

# Whitespace merge/split simulated via token joins/splits (lightweight)
def whitespace_merge(tokens: List[str], labels: List[int], p: float = 0.05) -> Tuple[List[str], List[int]]:
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
        if i < len(tokens) - 1 and random.random() < p:
            # merge tokens i and i+1
            out_tokens.append(tokens[i] + tokens[i+1])
            out_labels.append(labels[i])  # keep first label
            skip = True
        else:
            out_tokens.append(tokens[i])
            out_labels.append(labels[i])
    return out_tokens, out_labels


def syntactic_noise(
    tokens: List[str],
    labels: List[int],
    o_label: int = None,
    p: float = 0.1,
) -> Tuple[List[str], List[int]]:
    """
    Apply syntactic noise to ~p fraction of tokens.

    For each selected token, randomly choose one syntactic op:
      - punct_insert   (inserts punctuation after token)
      - punct_delete   (removes punctuation token)
      - whitespace_merge (merges token with next)
    """
    if not tokens:
        return tokens, labels

    n = len(tokens)
    k = max(0, int(round(n * p)))
    if k == 0:
        return tokens, labels

    change = set(random.sample(range(n), k))
    out_tokens, out_labels = [], []
    i = 0

    while i < len(tokens):
        if i in change:
            op = random.choice(["punct_insert", "punct_delete", "whitespace_merge"])

            if op == "punct_insert":
                puncts = [",", ".", ";", ":", "!", "?"]
                out_tokens.append(tokens[i])
                out_labels.append(labels[i])
                out_tokens.append(random.choice(puncts))
                out_labels.append(o_label)

            elif op == "punct_delete" and tokens[i] in ",.;:!?":
                # drop this token
                i += 1
                continue

            elif op == "whitespace_merge" and i < n - 1:
                out_tokens.append(tokens[i] + tokens[i + 1])
                out_labels.append(labels[i])  # keep label of first
                i += 2
                continue

            else:
                # fallback → keep original
                out_tokens.append(tokens[i])
                out_labels.append(labels[i])
        else:
            out_tokens.append(tokens[i])
            out_labels.append(labels[i])

        i += 1

    return out_tokens, out_labels

