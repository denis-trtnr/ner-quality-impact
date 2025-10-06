from nltk.corpus import wordnet

def penn_to_wordnet(penn_tag: str) -> str:
    """Converts Penn Treebank POS tags to WordNet compatible tags."""
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Fallback to noun if no clear mapping is found.
        return wordnet.NOUN