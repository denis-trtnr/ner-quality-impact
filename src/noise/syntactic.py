import random
from typing import List, Tuple, Dict, Callable

# Simple punctuation perturbations
def punct_insert(tokens: List[str], labels: List[int], o_label: int = None, p: float = 0.05) -> List[str]:
    """
    Insert random punctuation after tokens with probability `p`.
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
    Delete punctuation tokens with probability `p`.
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
    Merge two adjacent tokens into one with probability `p`.
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
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    o_label: int = None,
    p: float = 0.1,
    ops: List[Callable[[List[str], List[int], int], Tuple[List[str], List[int], int]]] = None,
) -> Tuple[List[str], List[int]]:
    """
    Apply syntactic noise to ~p fraction of tokens.

    For each selected token, randomly choose one syntactic op:
      - punct_insert_at        (inserts punctuation after token)
      - punct_delete_at        (removes punctuation token)
      - whitespace_merge_at    (merges token with next)
      - whitespace_split_at    (splits token into two at random position)
      - token_drop_at          (deletes token)
      - token_swap_adjacent_at (swaps token with next token)
    """
    if not tokens or p <= 0.0:
        return tokens, labels
    
    if ops is None:
        ops = [
            punct_insert_at,
            punct_delete_at,
            whitespace_merge_at,
            whitespace_split_at,
            token_drop_at,
            token_swap_adjacent_at,
        ]

    n = len(tokens)
    k = max(0, int(round(n * p)))
    if k == 0:
        return tokens, labels

    change = set(random.sample(range(n), k))
    additional_params = {"o_label": o_label, "id2label": id2label, "label2id": label2id}
    out_tokens, out_labels = tokens[:], labels[:]
    i = 0

    while i < len(tokens):
        apply_here = (i < n) and (i in change)

        if apply_here:
            op = random.choice(ops)
            out_tokens, out_labels, i = op(out_tokens, out_labels, i, **additional_params)
        else:
            i += 1
    return out_tokens, out_labels


def punct_insert_at(tokens: List[str], labels: List[int], i: int, **additional_params) -> Tuple[List[str], List[int], int]:
    """
    Insert a punctuation token after position i.
    The inserted punctuation always gets label 'O' (id = o_label).
    """
    if i >= len(tokens):
        return tokens, labels, i + 1
    puncts = [",", ".", ";", ":", "!", "?"]
    o_label = additional_params["o_label"]
    t, l = tokens[:], labels[:]
    t.insert(i + 1, random.choice(puncts))
    l.insert(i + 1, o_label)
    return t, l, i + 2

def punct_delete_at(tokens: List[str], labels: List[int], i: int, **additional_params) -> Tuple[List[str], List[int], int]:
    """Delete token i if it is punctuation; otherwise no-op."""
    if i >= len(tokens):
        return tokens, labels, i + 1
    if tokens[i] in ",.;:!?":
        t, l = tokens[:], labels[:]
        del t[i]; del l[i]
        return t, l, i
    return tokens, labels, i + 1

def whitespace_merge_at(tokens: List[str], labels: List[int], i: int, **additional_params) -> Tuple[List[str], List[int], int]:
    """Merge token i with token i+1 (keep label of the first)."""
    if i >= len(tokens):
        return tokens, labels, i + 1
    if i < len(tokens) - 1:
        t, l = tokens[:], labels[:]
        t[i] = t[i] + t[i + 1]
        del t[i + 1]; del l[i + 1]
        return t, l, i + 1
    return tokens, labels, i + 1

def whitespace_split_at(tokens: List[str], labels: List[int], i: int, **additional_params) -> Tuple[List[str], List[int], int]:
    """
    Split token i into two tokens at a random char boundary (len>=2).
    BIO rule: left keeps original; right becomes I-X if original was B/I-X; O->O|O.
    """
    if i >= len(tokens):
        return tokens, labels, i + 1
    id2label: Dict[int, str] = additional_params["id2label"]
    label2id: Dict[str, int] = additional_params["label2id"]

    tok = tokens[i]
    if len(tok) < 2:
        return tokens, labels, i + 1

    cut = random.randint(1, len(tok) - 1)
    left, right = tok[:cut], tok[cut:]

    t, l = tokens[:], labels[:]
    lab_id = l[i]
    lab_str = id2label[lab_id]

    t[i] = left
    if lab_str.startswith(("B-", "I-")):
        etype = lab_str.split("-", 1)[1]
        right_lab = label2id.get(f"I-{etype}", lab_id)
    else:
        right_lab = lab_id  # O

    t.insert(i + 1, right)
    l.insert(i + 1, right_lab)
    return t, l, i + 2

def token_drop_at(tokens: List[str], labels: List[int], i: int, **additional_params) -> Tuple[List[str], List[int], int]:
    """Drop token i and its label (avoid empty sequence)."""
    if i >= len(tokens):
        return tokens, labels, i + 1
    if len(tokens) <= 1:
        return tokens, labels, i + 1
    t, l = tokens[:], labels[:]
    del t[i]; del l[i]
    return t, l, i

def token_swap_adjacent_at(tokens: List[str], labels: List[int], i: int, **additional_params) -> Tuple[List[str], List[int], int]:
    """Swap tokens i and i+1 and labels."""
    if i >= len(tokens):
        return tokens, labels, i + 1
    if i < len(tokens) - 1:
        t, l = tokens[:], labels[:]
        t[i], t[i + 1] = t[i + 1], t[i]
        l[i], l[i + 1] = l[i + 1], l[i]
        return t, l, i + 2
    return tokens, labels, i + 1