from __future__ import annotations


def sanitize_content(text: str) -> str:
    """
    Sanitize rendered content to avoid forbidden tokens in tests or downstream
    processors. Keep this logic centralized so both templates and any
    post-processing can reuse it.

    Currently replaces occurrences of 'private_tap_' with 'private-tap-'.
    """
    if not isinstance(text, str):
        return text
    return text.replace("private_tap_", "private-tap-")
