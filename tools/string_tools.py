def word_count(text: str) -> str:
    """
    Count the number of words in a given text.

    Args:
        text: The input text to count words in

    Returns:
        Word count as a string
    """
    count = len(text.split())
    return f"The text contains {count} words"


def reverse_string(text: str) -> str:
    """
    Reverse the characters in a given string.

    Args:
        text: The string to reverse

    Returns:
        Reversed string
    """
    return text[::-1]