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
    """Returns adjacent characters on a QWERTZ keyboard."""
    return QWERTZ_NEIGHBORS.get(ch.lower(), "")