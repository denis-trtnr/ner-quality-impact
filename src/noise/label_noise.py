import random
from dataclasses import dataclass
from typing import List, Dict

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

def apply_label_noise_on_spans(ner_tags: List[int], id2label: Dict[int, str], label2id: Dict[str, int], p: float = 0.1, to_O_prob: float = 0.5) -> List[int]:
    labels = [id2label[i] for i in ner_tags]
    spans = extract_spans(labels)
    if not spans:
        return ner_tags
    n_change = max(0, int(round(len(spans) * p)))
    change_spans = set(random.sample(range(len(spans)), n_change))
    etypes = sorted({lab[2:] for lab in id2label.values() if lab.startswith("B-")})
    for idx in change_spans:
        s = spans[idx]
        if random.random() < to_O_prob:
            for k in range(s.start, s.end + 1):
                labels[k] = "O"
        else:
            choices = [t for t in etypes if t != s.etype]
            if not choices:
                continue
            new_type = random.choice(choices)
            labels[s.start] = f"B-{new_type}"
            for k in range(s.start + 1, s.end + 1):
                labels[k] = f"I-{new_type}"
    return [label2id[l] for l in labels]