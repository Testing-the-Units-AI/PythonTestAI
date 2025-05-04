def normalize_phone_number(phone: str) -> str:
    """
    Normalizes a phone number to the format +1-XXX-XXX-XXXX.

    - Accepts numbers with or without country code, dashes, spaces, or parentheses.
    - Raises ValueError if the number is invalid.

    Example:
        "(123) 456-7890" -> "+1-123-456-7890"
        "1234567890" -> "+1-123-456-7890"
    """
    import re

    digits = re.sub(r'\D', '', phone)

    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    elif len(digits) == 10:
        pass
    else:
        raise ValueError(f"Invalid phone number: {phone}")

    return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
