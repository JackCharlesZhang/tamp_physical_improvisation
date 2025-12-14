"""Base class for shortcut heuristics."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData, ObsType


class BaseHeuristic(ABC):
    """Base class for all shortcut heuristics.

    All heuristics follow a unified interface:
    1. multi_train() - Train the heuristic (or do nothing for non-learning methods)
    2. estimate_distance() - Estimate distance from state to node
    3. estimate_node_distance() - Estimate distance between nodes
    4. prune() - Filter training data to promising shortcuts

    This allows the pipeline to treat all heuristic methods uniformly.

    The training data and graph distances are stored at initialization and
    used by all methods.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
    ):
        """Initialize heuristic with data.

        Args:
            training_data: Full training data from collect_total_shortcuts
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
        """
        self.training_data = training_data
        self.graph_distances = graph_distances

    @abstractmethod
    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Train the heuristic on shortcut data.

        For non-learning methods (e.g., rollouts), this may do nothing.

        Args:
            **kwargs: Additional training parameters (epochs, rounds, etc.)

        Returns:
            Dictionary with training history/metadata
        """
        pass

    @abstractmethod
    def estimate_distance(self, state: "ObsType", target_node: int) -> float:
        """Estimate distance from a state to a target node.

        Args:
            state: Source state
            target_node: Target node ID

        Returns:
            Estimated distance (in steps)
        """
        pass

    @abstractmethod
    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Estimate distance between two nodes.

        Args:
            source_node: Source node ID
            target_node: Target node ID

        Returns:
            Estimated distance (in steps)
        """
        pass

    @abstractmethod
    def prune(self, **kwargs: Any) -> "GoalConditionedTrainingData":
        """Prune training data to promising shortcuts.

        Uses estimate_node_distance to evaluate shortcuts and filters
        to those worth training on.

        Args:
            **kwargs: Additional pruning parameters (threshold, keep_fraction, etc.)

        Returns:
            New GoalConditionedTrainingData with filtered shortcuts
        """
        pass

    def save(self, path: str) -> None:
        """Save heuristic to disk (optional, for learned heuristics)."""
        raise NotImplementedError("This heuristic does not support saving")

    def load(self, path: str) -> None:
        """Load heuristic from disk (optional, for learned heuristics)."""
        raise NotImplementedError("This heuristic does not support loading")
