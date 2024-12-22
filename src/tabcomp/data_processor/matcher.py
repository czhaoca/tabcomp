"""
Field matching module for TabComp.

This module provides functionality for intelligent field name matching between
tables, using various string similarity algorithms and heuristics to identify
corresponding columns across different data sources.

Example:
    >>> matcher = FieldMatcher()
    >>> result = matcher.match_fields(['customer_id', 'first_name'], 
                                    ['id_customer', 'firstName'])
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import re
from difflib import SequenceMatcher
from collections import defaultdict

from .exceptions import MatchingError


@dataclass
class MatchConfig:
    """Configuration for field matching operations."""

    case_sensitive: bool = False
    ignore_whitespace: bool = True
    match_threshold: float = 0.85
    exact_match_boost: float = 0.15
    common_prefix_boost: float = 0.1
    common_suffix_boost: float = 0.1
    acronym_match_boost: float = 0.2


class FieldMatcher:
    """
    Handles intelligent field name matching between tables.

    This class provides methods for matching field names using various
    string similarity algorithms and heuristics, with configurable
    matching criteria and thresholds.
    """

    # Common field name patterns to normalize
    COMMON_PATTERNS = {
        r"_id$": "id",
        r"[-_\s]": " ",
        r"([a-z])([A-Z])": r"\1 \2",  # Split camelCase
    }

    # Common field name prefixes/suffixes to handle specially
    COMMON_PREFIXES = {"customer", "user", "product", "order", "transaction"}
    COMMON_SUFFIXES = {"id", "code", "num", "number", "date", "amount"}

    def __init__(self, config: Optional[MatchConfig] = None):
        """
        Initialize the FieldMatcher with optional configuration.

        Args:
            config: Configuration options for field matching.
                   If not provided, uses default values.
        """
        self.config = config or MatchConfig()
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def match_fields(
        self,
        fields1: List[str],
        fields2: List[str],
        primary_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Match fields between two tables using similarity algorithms.

        Args:
            fields1: List of field names from first table
            fields2: List of field names from second table
            primary_key: Optional field to prioritize in matching
            **kwargs: Additional keyword arguments to override config

        Returns:
            Dictionary containing:
                - matched_fields: Dict mapping fields1 to fields2
                - unmatched_fields1: List of unmatched fields from fields1
                - unmatched_fields2: List of unmatched fields from fields2
                - match_confidence: Dict of confidence scores for matches
                - suggested_matches: Dict of potential matches below threshold

        Raises:
            MatchingError: If matching cannot be performed
            ValueError: If input field lists are invalid
        """
        try:
            # Validate inputs
            self._validate_inputs(fields1, fields2)

            # Update config with any provided overrides
            config = self._update_config(kwargs)

            # Normalize field names
            norm_fields1 = [self._normalize_field_name(f) for f in fields1]
            norm_fields2 = [self._normalize_field_name(f) for f in fields2]

            # Initialize results
            matches = {}
            unmatched1 = set(fields1)
            unmatched2 = set(fields2)
            confidence_scores = {}
            suggestions = defaultdict(list)

            # First pass: Find exact matches
            self._find_exact_matches(
                fields1,
                fields2,
                norm_fields1,
                norm_fields2,
                matches,
                unmatched1,
                unmatched2,
                confidence_scores,
            )

            # Second pass: Find high similarity matches
            self._find_similarity_matches(
                list(unmatched1),
                list(unmatched2),
                matches,
                unmatched1,
                unmatched2,
                confidence_scores,
                suggestions,
            )

            # Third pass: Try matching any remaining fields with relaxed threshold
            if unmatched1 and unmatched2:
                self._find_relaxed_matches(
                    list(unmatched1),
                    list(unmatched2),
                    matches,
                    unmatched1,
                    unmatched2,
                    confidence_scores,
                    suggestions,
                )

            return {
                "matched_fields": matches,
                "unmatched_fields1": list(unmatched1),
                "unmatched_fields2": list(unmatched2),
                "match_confidence": confidence_scores,
                "suggested_matches": dict(suggestions),
            }

        except Exception as e:
            raise MatchingError(f"Failed to match fields: {str(e)}")

    def calculate_similarity(
        self, field1: str, field2: str, normalized: bool = False
    ) -> float:
        """
        Calculate similarity score between two field names.

        Args:
            field1: First field name
            field2: Second field name
            normalized: Whether input fields are already normalized

        Returns:
            Similarity score between 0 and 1
        """
        # Check cache first
        cache_key = (field1, field2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Normalize if needed
        if not normalized:
            field1 = self._normalize_field_name(field1)
            field2 = self._normalize_field_name(field2)

        # Base similarity using sequence matcher
        similarity = SequenceMatcher(None, field1, field2).ratio()

        # Apply boosts for exact matches of components
        words1 = set(field1.split())
        words2 = set(field2.split())

        # Boost for exact word matches
        common_words = words1 & words2
        if common_words:
            similarity += self.config.exact_match_boost * (
                len(common_words) / max(len(words1), len(words2))
            )

        # Boost for common prefixes
        if any(p in words1 and p in words2 for p in self.COMMON_PREFIXES):
            similarity += self.config.common_prefix_boost

        # Boost for common suffixes
        if any(s in words1 and s in words2 for s in self.COMMON_SUFFIXES):
            similarity += self.config.common_suffix_boost

        # Boost for acronym matches
        acronym1 = "".join(w[0] for w in words1 if w)
        acronym2 = "".join(w[0] for w in words2 if w)
        if acronym1 == acronym2 and len(acronym1) > 1:
            similarity += self.config.acronym_match_boost

        # Cap at 1.0 and cache result
        similarity = min(1.0, similarity)
        self._similarity_cache[cache_key] = similarity
        return similarity

    def _normalize_field_name(self, field: str) -> str:
        """
        Normalize field name for comparison.

        Args:
            field: Field name to normalize

        Returns:
            Normalized field name
        """
        result = field

        # Apply case normalization if configured
        if not self.config.case_sensitive:
            result = result.lower()

        # Apply common pattern replacements
        for pattern, replacement in self.COMMON_PATTERNS.items():
            result = re.sub(pattern, replacement, result)

        # Remove whitespace if configured
        if self.config.ignore_whitespace:
            result = "".join(result.split())

        return result

    def _find_exact_matches(
        self,
        fields1: List[str],
        fields2: List[str],
        norm_fields1: List[str],
        norm_fields2: List[str],
        matches: Dict[str, str],
        unmatched1: Set[str],
        unmatched2: Set[str],
        confidence_scores: Dict[str, float],
    ) -> None:
        """
        Find and record exact matches between field names.

        Args:
            fields1: Original field names from first table
            fields2: Original field names from second table
            norm_fields1: Normalized field names from first table
            norm_fields2: Normalized field names from second table
            matches: Dict to store matched field pairs
            unmatched1: Set of unmatched fields from first table
            unmatched2: Set of unmatched fields from second table
            confidence_scores: Dict to store match confidence scores
        """
        for i, norm1 in enumerate(norm_fields1):
            try:
                j = norm_fields2.index(norm1)
                orig1, orig2 = fields1[i], fields2[j]
                matches[orig1] = orig2
                unmatched1.remove(orig1)
                unmatched2.remove(orig2)
                confidence_scores[orig1] = 1.0
            except ValueError:
                continue

    def _find_similarity_matches(
        self,
        remaining1: List[str],
        remaining2: List[str],
        matches: Dict[str, str],
        unmatched1: Set[str],
        unmatched2: Set[str],
        confidence_scores: Dict[str, float],
        suggestions: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """
        Find matches based on similarity scores above threshold.

        Args:
            remaining1: Unmatched fields from first table
            remaining2: Unmatched fields from second table
            matches: Dict to store matched field pairs
            unmatched1: Set of unmatched fields from first table
            unmatched2: Set of unmatched fields from second table
            confidence_scores: Dict to store match confidence scores
            suggestions: Dict to store suggested matches below threshold
        """
        # Calculate all pairwise similarities
        similarities = []
        for field1 in remaining1:
            for field2 in remaining2:
                similarity = self.calculate_similarity(field1, field2)
                if similarity >= self.config.match_threshold:
                    similarities.append((similarity, field1, field2))
                elif similarity >= self.config.match_threshold * 0.8:
                    suggestions[field1].append((field2, similarity))

        # Sort by similarity descending
        similarities.sort(reverse=True)

        # Match greedily, taking highest similarities first
        used1 = set()
        used2 = set()
        for similarity, field1, field2 in similarities:
            if field1 not in used1 and field2 not in used2:
                matches[field1] = field2
                unmatched1.remove(field1)
                unmatched2.remove(field2)
                confidence_scores[field1] = similarity
                used1.add(field1)
                used2.add(field2)

    def _find_relaxed_matches(
        self,
        remaining1: List[str],
        remaining2: List[str],
        matches: Dict[str, str],
        unmatched1: Set[str],
        unmatched2: Set[str],
        confidence_scores: Dict[str, float],
        suggestions: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """
        Find matches using relaxed threshold for remaining fields.

        Args:
            remaining1: Unmatched fields from first table
            remaining2: Unmatched fields from second table
            matches: Dict to store matched field pairs
            unmatched1: Set of unmatched fields from first table
            unmatched2: Set of unmatched fields from second table
            confidence_scores: Dict to store match confidence scores
            suggestions: Dict to store suggested matches below threshold
        """
        relaxed_threshold = self.config.match_threshold * 0.7

        for field1 in remaining1.copy():
            best_match = None
            best_similarity = relaxed_threshold

            for field2 in remaining2:
                similarity = self.calculate_similarity(field1, field2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = field2

            if best_match:
                matches[field1] = best_match
                unmatched1.remove(field1)
                unmatched2.remove(best_match)
                confidence_scores[field1] = best_similarity

    def _validate_inputs(self, fields1: List[str], fields2: List[str]) -> None:
        """
        Validate input field lists.

        Args:
            fields1: Field names from first table
            fields2: Field names from second table

        Raises:
            ValueError: If inputs are invalid
        """
        if not fields1 or not fields2:
            raise ValueError("Both field lists must be non-empty")

        if len(set(fields1)) != len(fields1):
            raise ValueError("Duplicate fields found in fields1")

        if len(set(fields2)) != len(fields2):
            raise ValueError("Duplicate fields found in fields2")

    def _update_config(self, kwargs: Dict[str, Any]) -> MatchConfig:
        """
        Update configuration with provided overrides.

        Args:
            kwargs: Dictionary of configuration overrides

        Returns:
            Updated configuration object
        """
        if not kwargs:
            return self.config

        # Create new config with original values
        updated_config = MatchConfig(
            case_sensitive=self.config.case_sensitive,
            ignore_whitespace=self.config.ignore_whitespace,
            match_threshold=self.config.match_threshold,
            exact_match_boost=self.config.exact_match_boost,
            common_prefix_boost=self.config.common_prefix_boost,
            common_suffix_boost=self.config.common_suffix_boost,
            acronym_match_boost=self.config.acronym_match_boost,
        )

        # Update with provided values
        for key, value in kwargs.items():
            if hasattr(updated_config, key):
                setattr(updated_config, key, value)

        return updated_config
