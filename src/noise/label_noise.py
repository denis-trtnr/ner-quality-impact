import random
from dataclasses import dataclass
from typing import List, Dict
from .utils import protect_token

@dataclass
class Span:
    start: int
    end: int
    etype: str

def extract_spans(labels: List[str]) -> List[Span]:
    spans = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            etype = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                j += 1
            spans.append(Span(i, j - 1, etype))
            i = j
        else:
            i += 1
    return spans


def apply_label_noise_on_spans(
    tokens: List[str],
    ner_tags: List[int],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    p: float,
    ops: List[str] = None,
    max_retries: int = 3,
) -> List[int]:
    """
    1. shorten entity span (shorten)
    2. extend entity span into following O's (extend)
    3. replace entire entity with 'O' (replace_O)
    4. replace with different entity class (other_class)
    5. turn random O-span into entity (token_to_entity)
    """
    # Fallback: use all operations if none specified
    if ops is None:
        ops = ["shorten", "extend", "replace_O", "other_class", "token_to_entity"]

    labels = [id2label[i] for i in ner_tags]
    spans = extract_spans(labels)
    if not spans:
        return ner_tags

    # determine how many entity spans will be affected
    n_change = max(1, int(round(len(spans) * p)))
    change_idxs = random.sample(range(len(spans)), n_change)
    etypes = sorted({lab[2:] for lab in id2label.values() if lab.startswith("B-")})

    # O-token indices (for potential new entities)
    O_idxs = [i for i, l in enumerate(labels) if l == "O" and not protect_token(tokens[i])]

    for idx in change_idxs:
        s = spans[idx]
        success = False

        for _ in range(max_retries):
            op = random.choice(ops)

            if op == "shorten" and (s.end - s.start) >= 1:
                labels[s.end] = "O"
                success = True
                break

            elif op == "extend" and s.end + 1 < len(labels) and labels[s.end + 1] == "O":
                if not protect_token(tokens[s.end + 1]):
                    labels[s.end + 1] = f"I-{s.etype}"
                    success = True
                    break

            elif op == "replace_O":
                for k in range(s.start, s.end + 1):
                    labels[k] = "O"
                success = True
                break

            elif op == "other_class":
                other_types = [t for t in etypes if t != s.etype]
                if other_types:
                    new_type = random.choice(other_types)
                    labels[s.start] = f"B-{new_type}"
                    for k in range(s.start + 1, s.end + 1):
                        labels[k] = f"I-{new_type}"
                    success = True
                    break

            elif op == "token_to_entity" and O_idxs:
                valid_O_idxs = [i for i in O_idxs if not protect_token(tokens[i])]
                if not valid_O_idxs:
                    continue
                i = random.choice(valid_O_idxs)
                O_idxs.remove(i)
                new_type = random.choice(etypes)
                labels[i] = f"B-{new_type}"
                # random extend
                if i + 1 < len(labels) and labels[i + 1] == "O" and not protect_token(tokens[i + 1]) and random.random() < 0.5:
                    labels[i + 1] = f"I-{new_type}"
                success = True
                break

        if not success:
            # Couldn’t apply any op
            continue

    return [label2id[l] for l in labels]