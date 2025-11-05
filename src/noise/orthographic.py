import random
import unicodedata
from typing import List, Dict
from .utils import neighbors, protect_token, DIACRITICS_CHAR_MAP, ASCII_HOMOGLYPHS

# Base typo ops
def swap_adjacent(word: str) -> str:
    # Swap two neighboring characters at a random position
    if len(word) < 3:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]

def delete_char(word: str) -> str:
    # Randomly delete one character (if long enough)
    if len(word) <= 1:
        return word
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]

def insert_char(word: str) -> str:
    # Insert a keyboard-neighbor character
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
    # Replace one character with a nearby key
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
    # Randomly flip upper/lower case of some letters
    return ''.join((c.lower() if c.isupper() else c.upper()) if random.random() < prob else c for c in word)

def strip_diacritics(word: str) -> str:
    # Unicode decomposition
    nfkd_form = unicodedata.normalize("NFKD", word)
    word = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Extended mapping
    word = word.translate(DIACRITICS_CHAR_MAP)
    word = unicodedata.normalize("NFKC", word)
    return word

def substitute_homoglyph(word: str, prob: float = 0.3) -> str:
    # Replace characters with visually similar ASCII homoglyphs
    if len(word) < 2:
        return word

    out = []
    for ch in word:
        if ch in ASCII_HOMOGLYPHS and random.random() < prob:
            repl = random.choice(ASCII_HOMOGLYPHS[ch])
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out)

# Compose
def typo_tokens(tokens: List[str], ner_tags: List[int], id2label: Dict[int, str], p: float, entity_strategy: str = "protect", ops=None) -> List[str]:
    """
    Apply orthographic (typo-level) noise to tokens.
    entity_strategy controls whether to protect or target entities:
      - 'protect'        -> do NOT modify entity tokens (default)
      - 'entities_only'  -> modify only entity tokens
      - 'all'            -> modify all tokens equally
    """
    if ops is None:
        ops = [
            swap_adjacent,
            delete_char,
            insert_char,
            substitute_char,
            strip_diacritics,
            random_case_flip,
            substitute_homoglyph
        ]
    # Collect candidate indices for modification
    idxs = []
    for i, (tok, tag_id) in enumerate(zip(tokens, ner_tags)):
        if protect_token(tok):
            continue

        lab = id2label[tag_id]
        is_entity = lab.startswith("B-") or lab.startswith("I-")
        if entity_strategy == "protect" and is_entity:
            continue
        if entity_strategy == "entities_only" and not is_entity:
            continue
        idxs.append(i)
    k = max(0, int(round(len(idxs) * p)))
    if k == 0:
        return tokens
    change = set(random.sample(idxs, k))
    out = []
    for i, tok in enumerate(tokens):
        # Apply a random typo operation to selected tokens
        if i in change:
            op = random.choice(ops)
            out.append(op(tok))
        else:
            out.append(tok)
    return out