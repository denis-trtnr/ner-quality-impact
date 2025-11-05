import random
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from .utils import (
    penn_to_wordnet,
    protect_token,
    load_static_embedding_model,
    load_contextual_embedding_model
)

def get_synonym_for_token(token: str, pos_tag: str, min_diff: float = 0.7) -> str:
    """Finds a synonym for a single token given its part-of-speech tag."""
    lemmatizer = WordNetLemmatizer()
    wn_pos = penn_to_wordnet(pos_tag)
    if not wn_pos:
        return token

    lemma = lemmatizer.lemmatize(token.lower(), pos=wn_pos)
    synsets = wordnet.synsets(lemma, pos=wn_pos)
    if not synsets:
        return token

    base_syn = synsets[0]  # use most frequent sense as reference
    candidates = set()

    for syn in synsets:
        sim = base_syn.wup_similarity(syn) or 0.0
        if sim < min_diff:  # keep only semantically distant synsets
            for l in syn.lemmas():
                cand = l.name().replace("_", " ")
                if cand.lower() != lemma:
                    candidates.add(cand)

    if not candidates:
        return token

    replacement = random.choice(list(candidates))
    if token.istitle():
        replacement = replacement.title()
    elif token.isupper():
        replacement = replacement.upper()
    return replacement

def get_word_embedding_for_token(token: str, model: Any) -> str:
    """Finds a replacement for a single token using static embeddings."""
    try:
        # Use`most_similar` from the loaded gensim model
        similar_words = model.most_similar(token.lower(), topn=30)
        candidates = [w for w, sim in similar_words if w.strip()]
        if len(candidates) > 10:
            # skip the top 10 most similar (too close)
            candidates = candidates[10:]
        if candidates:
            return random.choice(candidates)
    except KeyError: # Happens if the word is not in the vocabulary
        return token
    return token


def get_contextual_substitutions(new_tokens: List[str], original_tokens: List[str], indices: List[int], model_name: str) -> List[str]:
    """
    Handles the batch processing for all contextual substitutions.
    This is a dedicated helper to keep the main composer clean.
    """
    fill_masker = load_contextual_embedding_model(model_name)
    mask_token = fill_masker.tokenizer.mask_token
    
    # Create a batch of sentences, each with a different token masked
    batch_of_masked_sentences = [" ".join(original_tokens[:i] + [mask_token] + original_tokens[i+1:]) for i in indices]

    try:
        batch_results = fill_masker(batch_of_masked_sentences)
    except Exception:
        return new_tokens # Return the tokens unmodified if the model fails

    # The pipeline can return a list or a list of lists
    if batch_results and not isinstance(batch_results[0], list):
        batch_results = [batch_results]

    if batch_results:
        for result_group, i in zip(batch_results, indices):
            valid_preds = [
                p['token_str'] for p in result_group[5:30]  # skip top 5 to avoid identical/redundant tokens
                if p['token_str'].strip() and p['token_str'].strip().lower() != original_tokens[i].lower()
            ]
            if valid_preds:
                replacement = random.choice(valid_preds)
                # Preserve case
                if new_tokens[i].istitle(): replacement = replacement.title()
                elif new_tokens[i].isupper(): replacement = replacement.upper()
                new_tokens[i] = replacement
    
    return new_tokens

def get_antonym_for_token(token: str, pos_tag: str) -> str:
    """Finds an antonym for a given token using WordNet."""
    lemmatizer = WordNetLemmatizer()
    wn_pos = penn_to_wordnet(pos_tag)
    if not wn_pos:
        return token

    lemma = lemmatizer.lemmatize(token.lower(), pos=wn_pos)
    antonyms = set()

    for syn in wordnet.synsets(lemma, pos=wn_pos):
        for l in syn.lemmas():
            for ant in l.antonyms():
                antonyms.add(ant.name().replace("_", " "))

    if not antonyms:
        return token

    replacement = random.choice(list(antonyms))
    if token.istitle():
        replacement = replacement.title()
    elif token.isupper():
        replacement = replacement.upper()
    return replacement

def preserve_case(original: str, replacement: str) -> str:
    """Preserve the capitalization style of the original token."""
    if original.istitle():
        return replacement.title()
    elif original.isupper():
        return replacement.upper()
    return replacement

def semantic_noise(
    tokens: List[str], 
    pos_tags: List[str], 
    ner_tags: List[int], 
    id2label: Dict[int, str], 
    p: float, 
    ops: List[str] = None,
    entity_strategy: str = "protect",
    **kwargs
) -> List[str]:
    """
    Applies a mix of semantic operations.
    """
    if ops is None or len(ops) == 0:
        ops = ["synonym", "word_embs","antonym", "contextual"]
    
    """
    print(f"[semantic_noise] Applying semantic noise with ops={ops}, "
          f"p={p}, entity_strategy='{entity_strategy}'")
    """

    new_tokens = list(tokens)
    n = len(tokens)

    candidates = []
    weights = []

    for i, tok in enumerate(tokens):
        if protect_token(tok):
            continue
        
        label = id2label[ner_tags[i]]
        is_entity = label.startswith("B-") or label.startswith("I-")

        # --- Entity strategy control ---
        if entity_strategy == "protect" and is_entity:
            # Skip entities entirely
            continue
        if entity_strategy == "entities_only" and not is_entity:
            # Only allow entities to be candidates
            continue

        # Add to candidate list
        candidates.append(i)

        # Weight content words higher
        if pos_tags[i].startswith(("N", "V", "J", "R")):
            weights.append(3.0)
        elif pos_tags[i].startswith(("P", "C", "I", "D")):
            weights.append(1.5)
        else:
            weights.append(1.0)

    if not candidates:
        return tokens
    
    k = max(1, int(round(n * p)))
    k = min(k, len(candidates))  # avoid selecting more than available candidates

    weights = np.array(weights)
    probs = weights / weights.sum()

    chosen_candidates = np.random.choice(candidates, size=k, replace=False, p=probs)

    need_static_model = any(op in ["word_embs", "synonym", "antonym"] for op in ops)
    static_model = None
    if need_static_model:
        static_model = load_static_embedding_model(kwargs.get("model_path", "glove-wiki-gigaword-100"))

    # Decide which operation to use for each index BEFORE executing
    op_plan = {idx: random.choice(ops) for idx in chosen_candidates}
    grouped_ops = defaultdict(list)
    for idx, op_name in op_plan.items():
        grouped_ops[op_name].append(idx)

    
    if "synonym" in grouped_ops:
        for i in grouped_ops["synonym"]:
            replacement = get_synonym_for_token(new_tokens[i], pos_tags[i], min_diff=0.5)
            if replacement == new_tokens[i]:
                #Fallback: try embedding-based replacement if synonym failed
                replacement = get_word_embedding_for_token(new_tokens[i], static_model)
            new_tokens[i] = preserve_case(new_tokens[i], replacement)

    if "antonym" in grouped_ops:
        for i in grouped_ops["antonym"]:
            replacement = get_antonym_for_token(new_tokens[i], pos_tags[i])
            #Fallback: use embedding-based replacement if no antonym found
            if replacement == new_tokens[i] and static_model is not None:
                replacement = get_word_embedding_for_token(new_tokens[i], static_model)
            new_tokens[i] = preserve_case(new_tokens[i], replacement)
    
    if "word_embs" in grouped_ops:
        for i in grouped_ops["word_embs"]:
            replacement = get_word_embedding_for_token(new_tokens[i], static_model)
            new_tokens[i] = preserve_case(new_tokens[i], replacement)
            
    # Process the expensive contextual operation in a single, efficient batch
    if "contextual" in grouped_ops:
        new_tokens = get_contextual_substitutions(
            new_tokens=new_tokens,
            original_tokens=tokens,
            indices=grouped_ops["contextual"],
            model_name=kwargs.get("model_name", "albert-base-v2")
        )
    
    return new_tokens