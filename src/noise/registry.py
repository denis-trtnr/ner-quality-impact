from typing import Callable, Dict, Any
from .orthographic import typo_tokens, random_case_flip, strip_diacritics
from .semantic import semantic_noise
from .label_noise import apply_label_noise_on_spans
from .syntactic import punct_insert, punct_delete, whitespace_merge, syntactic_noise

# Registry maps string keys to callables
TOKEN_NOISE: Dict[str, Callable] = {
    "typo_tokens": typo_tokens,                     # args: tokens, ner_tags, id2label, p, protect_entities, ops?
    "random_case_flip": random_case_flip,           # args: word, prob
    "strip_diacritics": strip_diacritics,           # args: word
    "semantic_noise": semantic_noise,               # args: tokens, pos_tags, ner_tags, id2label, p, ops, entity_strategy, model_path
    "punct_insert": punct_insert,                   # args: tokens, prob
    "punct_delete": punct_delete,                   # args: tokens, prob
    "whitespace_merge": whitespace_merge,           # args: tokens, prob
    "syntactic_noise": syntactic_noise,             # args: tokens, labels, prob, o_label
}

LABEL_NOISE: Dict[str, Callable] = {
    "label_spans_uniform": apply_label_noise_on_spans, # args: ner_tags, id2label, label2id, p, to_O_prob
}