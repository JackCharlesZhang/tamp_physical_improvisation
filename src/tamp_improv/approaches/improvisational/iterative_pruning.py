"""This file contains functions that iteratively trains the distance heuristic, then
    uses the distance heuristic to prune the dataset, then uses the pruned dataset to train
    the distance heuristic again."""

from typing import Any

from tamp_improv.approaches.improvisational.pruning import (
    train_distance_heuristic,
    prune_with_distance_heuristic
)

import numpy as np

from tamp_improv.approaches.improvisational.distance_heuristic import (
    GoalConditionedDistanceHeuristic,
)
from tamp_improv.approaches.improvisational.pruning import prune_random
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


def iteratively_prune_training_data(
    training_data: GoalConditionedTrainingData,
    system: ImprovisationalTAMPSystem,
    planning_graph: PlanningGraph,
    config: dict[str, Any],
    rng: np.random.Generator,
    heuristic: "GoalConditionedDistanceHeuristic | None" = None,
):
    """This function should iterative train and then prune the training data for 
    max_pruning_iterations number of steps.
    
    Returns the trained heuristic.
    """

    max_pruning_iterations = config.get('max_pruning_iterations', 2)
    max_shortcuts = config.get("max_shortcuts_per_graph", 150)
    
    for _ in range(max_pruning_iterations):
        print(" FIRST TIME PRUNING TRAINING DATA")
        # First, train the distance heuristic
        heuristic = train_distance_heuristic(training_data, system, config, rng)

        # Given heuristic, prune the training data
        pruned_data = prune_with_distance_heuristic(
            training_data, system, planning_graph, config, rng, heuristic=heuristic
        )

        # Limit the number of training data to be max_shortcuts
        training_data = prune_random(pruned_data, max_shortcuts, rng)
    
    return heuristic

