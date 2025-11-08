"""
Simple tokenizer for the Parrot Model.

This module provides a basic tokenizer for counting tokens and
implementing token-based text limiting. It uses simple whitespace
splitting to avoid heavy dependencies like tiktoken.
"""


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.

    Uses simple whitespace-based tokenization. This is a lightweight
    approach that avoids dependencies on external tokenizers like tiktoken.

    Args:
        text: The text to count tokens in.

    Returns:
        The number of tokens (whitespace-separated words) in the text.

    Example:
        >>> count_tokens("Hello world")
        2
        >>> count_tokens("The quick brown fox")
        4
        >>> count_tokens("")
        0
    """
    if not text:
        return 0
    return len(text.split())


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens (whitespace-separated words).

    Since this uses whitespace-based tokenization, truncation naturally occurs
    at word boundaries.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens to keep.

    Returns:
        Truncated text with at most max_tokens tokens.

    Example:
        >>> truncate_to_tokens("The quick brown fox jumps", 3)
        'The quick brown'
        >>> truncate_to_tokens("Hello world", 5)
        'Hello world'
        >>> truncate_to_tokens("", 10)
        ''
    """
    if not text or max_tokens <= 0:
        return ""

    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return " ".join(tokens[:max_tokens])


def truncate_to_chars(text: str, max_chars: int, at_word: bool = True) -> str:
    """
    Truncate text to a maximum number of characters.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to keep.
        at_word: If True, truncate at the last complete word before max_chars.
            If False, truncate exactly at max_chars.

    Returns:
        Truncated text with at most max_chars characters.

    Example:
        >>> truncate_to_chars("The quick brown fox", 10, at_word=True)
        'The quick'
        >>> truncate_to_chars("The quick brown fox", 10, at_word=False)
        'The quick '
        >>> truncate_to_chars("Hello", 10)
        'Hello'
        >>> truncate_to_chars("", 10)
        ''
    """
    if not text or max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    if not at_word:
        # Simple truncation at exact character position
        return text[:max_chars]

    # Truncate at word boundary
    # Find the last space before or at max_chars
    truncated = text[:max_chars]

    # If we're in the middle of a word, find the last space
    last_space = truncated.rfind(" ")

    if last_space > 0:
        # Truncate at the last space
        return truncated[:last_space]
    else:
        # No space found, return the truncated text or empty if at_word is strict
        # We'll return at least something if there's no space
        return truncated
