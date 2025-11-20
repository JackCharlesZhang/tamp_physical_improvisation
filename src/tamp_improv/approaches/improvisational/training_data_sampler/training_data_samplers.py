from tamp_improv.approaches.improvisational.training_data_sampler.base_training_data_sampler import BaseTrainingDataSampler
from typing import TypeVar, Any, Optional
import numpy as np
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")

class RandomTrainingDataSampler(BaseTrainingDataSampler):
    """Sampler that selects state pairs randomly."""
    def __init__(self, 
        all_state_pairs: list[tuple[ObsType, ObsType]],
        system: ImprovisationalTAMPSystem, 
        planning_graph: Optional[PlanningGraph], 
        config: dict[str, Any], 
        rng: np.random.Generator,
    ):
        super().__init__(all_state_pairs, system, planning_graph, config, rng)

    def sample(self) -> list[tuple[ObsType, ObsType]]:
        num_train = min(self.config.get("num_training_pairs", 100), len(self.all_state_pairs))
        train_indices = self.rng.choice(len(self.all_state_pairs), size=num_train, replace=False)
        training_pairs = [self.all_state_pairs[i] for i in train_indices]
        return training_pairs


class MaxDistanceTrainingDataSampler(BaseTrainingDataSampler):
    """Sampler that selects state pairs with the greatest distance from each other.
    
    Computes Euclidean distance between flattened source and target states,
    then selects the top num_training_pairs pairs with the greatest distances.
    """
    
    def __init__(self, 
        all_state_pairs: list[tuple[ObsType, ObsType]],
        system: ImprovisationalTAMPSystem, 
        planning_graph: Optional[PlanningGraph], 
        config: dict[str, Any], 
        rng: np.random.Generator,
    ):
        super().__init__(all_state_pairs, system, planning_graph, config, rng)

    def _flatten_obs(self, obs: ObsType) -> np.ndarray:
        """Flatten observation to array for distance computation."""
        if hasattr(obs, "nodes"):
            return obs.nodes.flatten().astype(np.float32)
        return np.array(obs).flatten().astype(np.float32)

    def sample(self) -> list[tuple[ObsType, ObsType]]:
        """Sample training pairs with greatest distance between source and target."""
        num_train = min(self.config.get("num_training_pairs", 100), len(self.all_state_pairs))
        
        # Compute distances for all pairs
        distances = []
        for source_state, target_state in self.all_state_pairs:
            source_flat = self._flatten_obs(source_state)
            target_flat = self._flatten_obs(target_state)
            # Compute Euclidean distance
            distance = np.linalg.norm(source_flat - target_flat)
            distances.append(distance)
        
        # Get indices sorted by distance (descending)
        sorted_indices = np.argsort(distances)[::-1]
        
        # Select top num_train pairs with greatest distances
        selected_indices = sorted_indices[:num_train]
        training_pairs = [self.all_state_pairs[i] for i in selected_indices]
        
        return training_pairs