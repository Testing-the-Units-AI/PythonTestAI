def parse_bool(val: str) -> bool:
    """
    Converts a string to a boolean. Accepts 'true', 'false', '1', '0', case-insensitive.
    Raises ValueError on unrecognized input.
    """
    val_lower = val.strip().lower()
    if val_lower in ('true', '1'):
        return True
    elif val_lower in ('false', '0'):
        return False
    else:
        raise ValueError(f"Cannot parse boolean from '{val}'")
