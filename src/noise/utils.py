import random

def set_py_random(seed: int):
    random.seed(seed)

# Keyboard neighborhoods (QWERTZ)
QWERTZ_NEIGHBORS = {
    "a": "qswy", "b": "vghn", "c": "xdfv", "d": "erfcxs", "e": "rdsw",
    "f": "rtgvcd", "g": "tyhbvf", "h": "uynjbg", "i": "uokj", "j": "uihnkm",
    "k": "ijolm", "l": "okp", "m": "njk", "n": "bhjm", "o": "iplö",
    "p": "olü", "q": "was", "r": "etdf", "s": "awedxz", "t": "ryfg",
    "u": "ziyhj", "v": "cfgb", "w": "qeas", "x": "zsdc", "y": "tugh",
    "z": "yuasx", "ä": "öü", "ö": "äü", "ü": "öä"
}

def neighbors(ch: str) -> str:
    return QWERTZ_NEIGHBORS.get(ch.lower(), "")

def is_punct(tok: str) -> bool:
    return all(not c.isalnum() for c in tok)

def protect_token(tok: str) -> bool:
    if len(tok) <= 3:
        return True
    if any(c.isdigit() for c in tok):
        return True
    if is_punct(tok):
        return True
    return False