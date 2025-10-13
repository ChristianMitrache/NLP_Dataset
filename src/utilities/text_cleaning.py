"""
Heuristic Utilities for text cleaning.
"""

import re
from unidecode import unidecode
from typing import Optional


def clean_string(item: Optional[str]):
    """
    Utility function for some general heuristic string cleaning
    """
    if not item:
        return ""
    item = str(item)
    item = unidecode(item)  # café → cafe, München → Munchen
    item = item.lower()
    item = item.replace("-", " ").replace("_", " ")
    item = re.sub(r"[^\w\s]", "", item)
    item = re.sub(r"\s+", " ", item)
    item = item.strip()
    return item
