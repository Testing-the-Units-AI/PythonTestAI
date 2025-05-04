def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flattens a nested dictionary.
    Example:
        {"a": {"b": 2}} -> {"a.b": 2}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items
