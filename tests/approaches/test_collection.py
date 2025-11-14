"""Tests for collection module."""

import pytest

from tamp_improv.approaches.improvisational.collection import (
    collect_all_shortcuts,
    collect_shortcuts_single_episode,
)


def test_collection_imports():
    """Test that collection functions can be imported."""
    assert callable(collect_all_shortcuts)
    assert callable(collect_shortcuts_single_episode)


# Note: Full integration tests would require setting up a real TAMP system
# and approach, which is more appropriate for integration tests.
# For now, we just verify the functions exist and are callable.
