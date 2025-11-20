from tamp_improv.approaches.improvisational.training_data_sampler.base_training_data_sampler import BaseTrainingDataSampler
from typing import TypeVar, Any
import numpy as np
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")

class RandomTrainingDataSampler(BaseTrainingDataSampler):
    def __init__(self, 
        all_state_pairs: list[tuple[ObsType, ObsType]],
        system: ImprovisationalTAMPSystem, 
        planning_graph: PlanningGraph, 
        config: dict[str, Any], 
        rng: np.random.Generator,
    ):
        super().__init__(all_state_pairs, system, planning_graph, config, rng)

    def sample(self) -> list[tuple[ObsType, ObsType]]:
        num_train = min(self.config.get("num_training_pairs", 100), len(self.all_state_pairs))
        train_indices = self.rng.choice(len(self.all_state_pairs), size=num_train, replace=False)
        training_pairs = [self.all_state_pairs[i] for i in train_indices]
        return training_pairs