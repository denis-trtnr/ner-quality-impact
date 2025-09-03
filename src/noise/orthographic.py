import random
from typing import List, Dict
from .utils import neighbors, protect_token

# Base typo ops
def swap_adjacent(word: str) -> str:
    if len(word) < 4:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]

def delete_char(word: str) -> str:
    if len(word) <= 3:
        return word
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]

def insert_char(word: str) -> str:
    i = random.randint(0, len(word) - 1)
    ch = word[i]
    nbrs = neighbors(ch)
    if not nbrs:
        return word
    ins = random.choice(list(nbrs))
    if ch.isupper():
        ins = ins.upper()
    return word[:i] + ins + word[i:]

def substitute_char(word: str) -> str:
    cands = [i for i, ch in enumerate(word) if neighbors(ch)]
    if not cands:
        return word
    i = random.choice(cands)
    ch = word[i]
    rep = random.choice(list(neighbors(ch)))
    if ch.isupper():
        rep = rep.upper()
    return word[:i] + rep + word[i + 1:]

def random_case_flip(word: str, prob: float = 0.3) -> str:
    return ''.join((c.lower() if c.isupper() else c.upper()) if random.random() < prob else c for c in word)

def strip_diacritics(word: str) -> str:
    table = str.maketrans({"ä":"a","ö":"o","ü":"u","Ä":"A","Ö":"O","Ü":"U","ß":"ss"})
    return word.translate(table)

# Compose
def typo_tokens(tokens: List[str], ner_tags: List[int], id2label: Dict[int, str], p: float, protect_entities: bool, ops=None) -> List[str]:
    if ops is None:
        ops = [swap_adjacent, delete_char, insert_char, substitute_char]
    idxs = []
    for i, (tok, tag_id) in enumerate(zip(tokens, ner_tags)):
        if protect_entities:
            lab = id2label[tag_id]
            if lab.startswith("B-") or lab.startswith("I-"):
                continue
        if protect_token(tok):
            continue
        idxs.append(i)
    k = max(0, int(round(len(idxs) * p)))
    if k == 0:
        return tokens
    change = set(random.sample(idxs, k))
    out = []
    for i, tok in enumerate(tokens):
        if i in change:
            op = random.choice(ops)
            out.append(op(tok))
        else:
            out.append(tok)
    return out

# Token drop/swap (syntactic-ish?)
def token_drop(tokens: List[str], p_drop: float) -> List[str]:
    out = []
    for t in tokens:
        if random.random() < p_drop:
            continue
        out.append(t)
    return out if out else tokens # avoid empty

def token_swap_adjacent(tokens: List[str], p_swap: float) -> List[str]:
    i = 0
    out = tokens[:]
    while i < len(out) - 1:
        if random.random() < p_swap:
            out[i], out[i+1] = out[i+1], out[i]
            i += 2
        else:
            i += 1
    return out