"""Data loading utilities for the conservation agent."""

import json
import os
from pathlib import Path
from typing import Any, Dict


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"


def load_species_data() -> Dict[str, Any]:
    """
    Load species data from JSON file.

    Returns:
        Dictionary containing species information.
    """
    data_path = get_data_dir() / "species_data.json"
    with open(data_path, "r") as f:
        return json.load(f)


def load_conservation_report() -> str:
    """
    Load the conservation report markdown file.

    Returns:
        String contents of the conservation report.
    """
    report_path = get_data_dir() / "animal_conservation_report.md"
    with open(report_path, "r") as f:
        return f.read()


def fuzzy_match_species(query: str, threshold: float = 0.6) -> str | None:
    """
    Find the best matching species name from the database.

    Args:
        query: User-provided species name.
        threshold: Minimum similarity score (0-1).

    Returns:
        The best matching species key or None if no match found.
    """
    data = load_species_data()
    query_lower = query.lower().strip()

    # First try exact match
    if query_lower in data:
        return query_lower

    # Try partial match (query is contained in key or vice versa)
    for key in data:
        if query_lower in key or key in query_lower:
            return key

    # Try word-based matching
    query_words = set(query_lower.split())
    best_match = None
    best_score = 0

    for key in data:
        key_words = set(key.split())
        intersection = query_words & key_words
        union = query_words | key_words

        if union:
            score = len(intersection) / len(union)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = key

    return best_match


def get_species_info(species_name: str) -> Dict[str, Any] | None:
    """
    Get species information with fuzzy matching.

    Args:
        species_name: Name of the species to look up.

    Returns:
        Species information dict or None if not found.
    """
    matched_key = fuzzy_match_species(species_name)
    if matched_key:
        data = load_species_data()
        return data[matched_key]
    return None
