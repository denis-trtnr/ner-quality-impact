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

DIACRITICS_CHAR_MAP = str.maketrans({
    "ß": "ss", "ẞ": "SS", "ä": "a", "ö": "o", "ü": "u",
    "Ä": "A", "Ö": "O", "Ü": "U",
    "ø": "o", "Ø": "O", "å": "a", "Å": "A", "æ": "ae", "Æ": "Ae",
    "œ": "oe", "Œ": "Oe",
    "ñ": "n", "Ñ": "N", "ç": "c", "Ç": "C",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "É": "E", "È": "E", "Ê": "E", "Ë": "E",
    "á": "a", "à": "a", "â": "a", "ã": "a", "å": "a", "ä": "a",
    "Á": "A", "À": "A", "Â": "A", "Ã": "A", "Å": "A", "Ä": "A",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "Í": "I", "Ì": "I", "Î": "I", "Ï": "I",
    "ó": "o", "ò": "o", "ô": "o", "õ": "o", "ö": "o",
    "Ó": "O", "Ò": "O", "Ô": "O", "Õ": "O", "Ö": "O",
    "ú": "u", "ù": "u", "û": "u", "ü": "u",
    "Ú": "U", "Ù": "U", "Û": "U", "Ü": "U",
    "ý": "y", "ÿ": "y", "Ý": "Y",
    "ł": "l", "Ł": "L", "š": "s", "Š": "S", "ž": "z", "Ž": "Z",
    "č": "c", "Č": "C", "ć": "c", "Ć": "C", "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th",
    "’": "'", "‘": "'", "´": "'", "`": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",  # dash variants
    "…": "...", "•": "*", "·": ".", "‚": ",", "¡": "!", "¿": "?",
})

ASCII_HOMOGLYPHS = {
    # Digits ↔ Letters / Symbols
    "0": ["O", "o", "8", "9", "D"],
    "1": ["l", "I", "7", "4"],
    "2": ["Z", "z"],
    "3": ["E", "8"],
    "4": ["A", "1"],
    "5": ["S", "8"],
    "6": ["G", "b"],
    "7": ["T", "1"],
    "8": ["B", "S", "@", "&"],
    "9": ["g", "q"],

    # Letters ↔ Digits / Similar Shapes
    "A": ["4"],
    "B": ["8"],
    "C": ["G", "O"],
    "D": ["0"],
    "E": ["3", "8"],
    "G": ["6", "C"],
    "I": ["1", "l"],
    "L": ["1", "I"],
    "O": ["0", "Q", "D"],
    "Q": ["O", "0"],
    "S": ["5", "8"],
    "T": ["7"],
    "Z": ["2"],

    # Lowercase variants / OCR confusions
    "a": ["o"],
    "b": ["6"],
    "c": ["e"],
    "d": ["cl"],
    "e": ["c"],
    "g": ["9", "q"],
    "i": ["l"],
    "l": ["1", "I"],
    "m": ["rn"],
    "n": ["r"],
    "o": ["0", "u"],
    "r": ["n"],
    "s": ["5"],
    "u": ["o"],
    "v": ["u"],
    "w": ["vv"],
}