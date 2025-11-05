def is_punct(tok: str) -> bool:
    """Determines if a token is punctuation."""
    return all(not c.isalnum() for c in tok)

def protect_token(tok: str) -> bool:
    """Determines if a token should be protected from augmentation."""
    if len(tok) <= 2: #short words like 'a', 'is', 'in'
        return True
    #if any(c.isdigit() for c in tok): # Protecting tokens with numbers
    #    return True
    if is_punct(tok): # Protecting punctuation
        return True
    return False