def calculator(operation: str, a: float, b: float) -> str:
    """
    Performs basic arithmetic operations.

    Args:
        operation: One of: add, subtract, multiply, divide
        a: First number
        b: Second number

    Returns:
        Result as string
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero"
        return str(a / b)
    return f"Error: unknown operation {operation}"