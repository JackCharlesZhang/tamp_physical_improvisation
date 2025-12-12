"""Utility classes for PRBench integration."""

from relational_structs import Predicate


class PRBenchPredicateContainer:
    """Adapts PRBench's predicate set to SLAP's predicate container protocol.

    SLAP expects predicates to be accessible via:
    - predicates["PredicateName"] -> Predicate
    - predicates.as_set() -> set[Predicate]

    PRBench provides predicates as a simple set, so we need this wrapper.

    Example:
        >>> from prbench_bilevel_planning.env_models.dynamic2d.dynobstruction2d import (
        ...     create_bilevel_planning_models
        ... )
        >>> sesame_models = create_bilevel_planning_models(obs_space, action_space, 2)
        >>> container = PRBenchPredicateContainer(sesame_models.predicates)
        >>> hand_empty = container["HandEmpty"]
        >>> all_preds = container.as_set()
    """

    def __init__(self, predicates: set[Predicate]) -> None:
        """Initialize container with PRBench's predicate set.

        Args:
            predicates: Set of predicates from PRBench's SesameModels
        """
        self._predicates = {p.name: p for p in predicates}

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name.

        Args:
            key: Predicate name (e.g., "HandEmpty", "HoldingTgt")

        Returns:
            Predicate object

        Raises:
            KeyError: If predicate name not found
        """
        return self._predicates[key]

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates.

        Returns:
            Set of all predicates
        """
        return set(self._predicates.values())

    def __contains__(self, key: str) -> bool:
        """Check if predicate exists.

        Args:
            key: Predicate name

        Returns:
            True if predicate exists, False otherwise
        """
        return key in self._predicates

    def __len__(self) -> int:
        """Get number of predicates.

        Returns:
            Number of predicates in container
        """
        return len(self._predicates)

    def __repr__(self) -> str:
        """String representation."""
        return f"PRBenchPredicateContainer({sorted(self._predicates.keys())})"
