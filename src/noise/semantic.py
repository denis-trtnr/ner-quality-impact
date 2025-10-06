import random
from typing import List, Dict, Any
from collections import defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from .utils import (
    penn_to_wordnet,
    protect_token,
    load_static_embedding_model,
    load_contextual_embedding_model
)

def get_synonym_for_token(token: str, pos_tag: str) -> str:
    """Finds a synonym for a single token given its part-of-speech tag."""
    lemmatizer = WordNetLemmatizer()
    wordnet_pos = penn_to_wordnet(pos_tag)
    
    synonyms = set()
    lemma = lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
    
    for syn in wordnet.synsets(lemma, pos=wordnet_pos):
        for syn_lemma in syn.lemmas():
            synonym = syn_lemma.name().replace('_', ' ')
            if synonym.lower() != lemma and ' ' not in synonym:
                synonyms.add(synonym)
    
    if synonyms:
        return random.choice(list(synonyms))
    return token

def get_word_embedding_for_token(token: str, model: Any) -> str:
    """Finds a replacement for a single token using static embeddings."""
    try:
        # Use`most_similar` from the loaded gensim model
        similar_words = model.most_similar(token.lower(), topn=5)
        candidates = [w for w, _ in similar_words if w.isalpha()]
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
                p['token_str'] for p in result_group 
                if p['token_str'].strip().lower() != original_tokens[i].lower() and p['token_str'].isalpha()
            ]
            if valid_preds:
                replacement = random.choice(valid_preds)
                # Preserve case
                if new_tokens[i].istitle(): replacement = replacement.title()
                elif new_tokens[i].isupper(): replacement = replacement.upper()
                new_tokens[i] = replacement
    
    return new_tokens

def semantic_noise(
    tokens: List[str], 
    pos_tags: List[str], 
    ner_tags: List[int], 
    id2label: Dict[int, str], 
    p: float, 
    ops: List[str],
    entity_strategy: str = "protect",
    **kwargs
) -> List[str]:
    """
    Applies a mix of semantic operations.
    """
    candidate_idxs = []
    for i, tok in enumerate(tokens):
        # 1. Apply general token protection
        if protect_token(tok):
            continue

        # 2. Apply entity-based strategy
        is_entity = id2label[ner_tags[i]].startswith("B-") or id2label[ner_tags[i]].startswith("I-")
        if entity_strategy == "protect" and is_entity:
            continue
        if entity_strategy == "entities_only" and not is_entity:
            continue
        
        # 3. Apply part-of-speech filter (only augment content words - Nouns, verbs, adjectives and adverbs)
        if pos_tags[i].startswith(("N", "V", "J", "R")):
             candidate_idxs.append(i)

    k = max(0, int(round(len(candidate_idxs) * p)))
    if k == 0 or not ops:
        return tokens
    
    change_indices = random.sample(candidate_idxs, k)
    new_tokens = list(tokens)

    # Decide which operation to use for each index BEFORE executing
    op_plan = {idx: random.choice(ops) for idx in change_indices}
    grouped_ops = defaultdict(list)
    for idx, op_name in op_plan.items():
        grouped_ops[op_name].append(idx)

    
    if "synonym" in grouped_ops:
        for i in grouped_ops["synonym"]:
            replacement = get_synonym_for_token(new_tokens, pos_tags, i)
            if new_tokens[i].istitle(): replacement = replacement.title()
            elif new_tokens[i].isupper(): replacement = replacement.upper()
            new_tokens[i] = replacement
    
    if "word_embs" in grouped_ops:
        model = load_static_embedding_model(kwargs.get("model_path", "glove-wiki-gigaword-100"))
        for i in grouped_ops["word_embs"]:
            replacement = get_word_embedding_for_token(new_tokens, i, model)
            if new_tokens[i].istitle(): replacement = replacement.title()
            elif new_tokens[i].isupper(): replacement = replacement.upper()
            new_tokens[i] = replacement
            
    # Process the expensive contextual operation in a single, efficient batch
    if "contextual" in grouped_ops:
        new_tokens = get_contextual_substitutions(
            new_tokens=new_tokens,
            original_tokens=tokens,
            indices=grouped_ops["contextual"],
            model_name=kwargs.get("model_name", "albert-base-v2")
        )
    
    return new_tokens