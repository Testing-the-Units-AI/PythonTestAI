def fibonacci(n: int) -> int:
    """
    Returns the nth Fibonacci number.
    Raises ValueError if n is negative.
    """
    if n < 0:
        raise ValueError("Input must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
