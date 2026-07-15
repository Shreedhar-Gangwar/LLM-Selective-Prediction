"""Tests for the intent grouping (src/groups.py)."""
from __future__ import annotations

from src.data import canonical_labels
from src.groups import GROUP_NAMES, build_groups, group_of


def test_every_intent_grouped_exactly_once():
    groups = build_groups()
    flat = [label for members in groups.values() for label in members]
    assert len(flat) == 77
    assert set(flat) == set(canonical_labels())  # exact cover, no duplicates


def test_group_of_is_total_and_stable():
    # Every canonical label maps to a known group name.
    for label in canonical_labels():
        assert group_of(label) in GROUP_NAMES


def test_override_applies():
    # contactless has no 'card' token but is explicitly filed under cards.
    assert group_of("contactless_not_working") == "cards"
