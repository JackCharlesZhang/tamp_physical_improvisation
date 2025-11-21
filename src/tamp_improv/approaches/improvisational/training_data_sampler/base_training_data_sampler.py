from typing import Any, TypeVar, Generic
import numpy as np
import abc
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


ObsType = TypeVar("ObsType")

class BaseTrainingDataSampler(Generic[ObsType], abc.ABC):

    def __init__(self, 
        training_data: GoalConditionedTrainingData,
        system: ImprovisationalTAMPSystem,
        config: dict[str, Any],
        rng: np.random.Generator,
    ):
        self.training_data = training_data
        self.system = system
        self.perceiver = self.system.perceiver
        self.config = config
        self.rng = rng

        # Prepare all state pairs from training data
        self.all_state_pairs = []
        for source_id, target_id in self.training_data.valid_shortcuts:
            if source_id not in self.training_data.node_states:
                continue
            if target_id not in self.training_data.node_states:
                continue

            source_states = self.training_data.node_states[source_id]
            target_states = self.training_data.node_states[target_id]

            if not source_states or not target_states:
                continue

            # Use all combinations of states
            for source_state in source_states:
                for target_state in target_states:
                    self.all_state_pairs.append((source_state, target_state))

        print(f"  Total available state pairs: {len(self.all_state_pairs)}")

    @abc.abstractmethod
    def sample(self) -> list[tuple[ObsType, ObsType]]:
        """Sample training pairs for distance heuristic training."""
        pass
