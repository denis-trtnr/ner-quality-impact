import random
from typing import List, Dict


# Minimal synonym substitution using a small hand-made map (later try looking into list or KG synonyms)
SYN_MAP = {
    "said": ["stated", "remarked"],
    "buy": ["purchase", "acquire"],
    "big": ["large", "huge"],
}

def synonym_substitute(tokens: List[str], ner_tags: List[int], id2label: Dict[int, str], p: float = 0.05) -> List[str]:
    out = []
    for tok, tag_id in zip(tokens, ner_tags):
        lab = id2label[tag_id]
        if lab.startswith("B-") or lab.startswith("I-"):
            out.append(tok)
            continue
        low = tok.lower()
        if low in SYN_MAP and random.random() < p:
            cand = random.choice(SYN_MAP[low])
            out.append(cand.capitalize() if tok[0].isupper() else cand)
        else:
            out.append(tok)
    return out