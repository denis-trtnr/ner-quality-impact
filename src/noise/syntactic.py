import random
from typing import List

# Simple punctuation perturbations
def punct_insert(text_tokens: List[str], prob: float = 0.05) -> List[str]:
    puncts = [",", ".", ";", ":", "!", "?"]
    out = []
    for t in text_tokens:
        out.append(t)
        if random.random() < prob:
            out.append(random.choice(puncts))
    return out

def punct_delete(text_tokens: List[str], prob: float = 0.1) -> List[str]:
    return [t for t in text_tokens if not (t in ",.;:!?" and random.random() < prob)]

# Whitespace merge/split simulated via token joins/splits (lightweight)
def whitespace_merge(tokens: List[str], prob: float = 0.05) -> List[str]:
    out = []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        if i < len(tokens)-1 and random.random() < prob:
            out.append(tokens[i] + tokens[i+1])
            skip = True
        else:
            out.append(tokens[i])
    return out