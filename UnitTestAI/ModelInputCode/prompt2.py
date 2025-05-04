def is_palindrome(s: str) -> bool:
    """
    Returns True if the input string is a palindrome, ignoring case and non-alphanumeric characters.
    """
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    return cleaned == cleaned[::-1]
