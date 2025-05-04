def calculate_discounted_price(price: float, discount: float) -> float:
    """
    Applies a discount (as a percentage) to the price and returns the new price.
    Raises ValueError if price or discount is negative, or discount is over 100%.
    """
    if price < 0 or discount < 0 or discount > 100:
        raise ValueError("Invalid price or discount")
    return round(price * (1 - discount / 100), 2)
