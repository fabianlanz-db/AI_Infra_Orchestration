"""Shared text extraction utilities used by skill registry, router, and judges."""

from __future__ import annotations

import re

STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "was",
    "one", "our", "out", "has", "have", "been", "from", "this", "that", "with",
    "they", "will", "each", "make", "like", "than", "them", "then", "into", "over",
    "such", "when", "very", "some", "just", "also", "more", "other", "would",
    "about", "should", "these", "their", "which", "could", "does", "most", "what",
    "only",
})


def extract_terms(text: str) -> set[str]:
    """Extract meaningful terms (3+ alpha chars, lowered, stop-words removed)."""
    return {t for t in re.findall(r"[a-zA-Z]{3,}", text.lower()) if t not in STOP_WORDS}
