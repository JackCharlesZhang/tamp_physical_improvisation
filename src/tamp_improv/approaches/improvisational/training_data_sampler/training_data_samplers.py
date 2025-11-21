from typing import TypeVar, Any, Optional
import numpy as np
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.approaches.improvisational.base import ShortcutSignature
from tamp_improv.approaches.improvisational.training_data_sampler.base_training_data_sampler import BaseTrainingDataSampler


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
    
class SimilarityTrainingDataSampler(BaseTrainingDataSampler):
    """Sampler that selects state pairs with minimum collective similarity.
    
    Uses greedy farthest-point sampling to select num_training_pairs state pairs
    such that the collective similarity between all selected pairs is minimized.
    This ensures maximum diversity in the training set.
    """
    
    def __init__(self, 
        all_state_pairs: list[tuple[ObsType, ObsType]],
        system: ImprovisationalTAMPSystem, 
        planning_graph: Optional[PlanningGraph], 
        config: dict[str, Any], 
        rng: np.random.Generator,
    ):
        super().__init__(all_state_pairs, system, planning_graph, config, rng)
    
    def sample(self) -> list[tuple[ObsType, ObsType]]:
        """Sample training pairs with minimum collective similarity using greedy selection.
        
        Algorithm: Greedy farthest-point sampling
        1. Start with a random pair
        2. Iteratively add the pair that has the minimum maximum similarity 
           to any already selected pair
        3. This ensures maximum diversity in the selected set
        """
        num_train = min(self.config.get("num_training_pairs", 100), len(self.all_state_pairs))

        if num_train == 0:
            return []
        
        assert self.planning_graph is not None, "Planning graph is required for similarity sampling"

        # Step 1: Map each state pair to its shortcut signature
        # We need to find which nodes correspond to each state pair
        pair_signatures = []
        assert self.planning_graph is not None
        for source_node in self.planning_graph.nodes:
            for target_node in self.planning_graph.nodes:
                if source_node == target_node:
                    continue
                if any(
                    edge.target == target_node
                    for edge in self.planning_graph.node_to_outgoing_edges.get(source_node, [])
                ):
                    continue
                if target_node.id <= source_node.id:
                    continue

                source_atoms = set(source_node.atoms)
                target_atoms = set(target_node.atoms)

                current_signature = ShortcutSignature.from_context(source_atoms, target_atoms)
                pair_signatures.append(current_signature)
        
        
        # Step 2: Greedy farthest-point sampling
        # Start with a random pair
        selected_indices: list[int] = []
        remaining_indices = set(range(len(self.all_state_pairs)))
        
        # Select first pair randomly
        first_idx = self.rng.choice(len(self.all_state_pairs))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select the pair with minimum maximum similarity to selected pairs
        for _ in range(num_train - 1):
            if not remaining_indices:
                break
            
            best_idx = None
            best_min_similarity = float('inf')
            
            # For each remaining pair, find its maximum similarity to any selected pair
            for candidate_idx in remaining_indices:
                candidate_sig = pair_signatures[candidate_idx]
                
                # Find maximum similarity to any already selected pair
                max_sim_to_selected = 0.0
                for selected_idx in selected_indices:
                    selected_sig = pair_signatures[selected_idx]
                    sim = candidate_sig.similarity(selected_sig)
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # We want the pair with minimum maximum similarity (most diverse)
                # So we maximize (1 - similarity) which is equivalent to minimizing similarity
                if max_sim_to_selected < best_min_similarity:
                    best_min_similarity = max_sim_to_selected
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Return selected pairs
        training_pairs: list[tuple[ObsType, ObsType]] = [self.all_state_pairs[i] for i in selected_indices]
        
        # Compute average pairwise similarity for reporting
        if len(selected_indices) > 1:
            total_sim = 0.0
            count = 0
            for i, idx1 in enumerate(selected_indices):
                for idx2 in selected_indices[i+1:]:
                    sim = pair_signatures[idx1].similarity(pair_signatures[idx2])
                    total_sim += sim
                    count += 1
            avg_sim = total_sim / count if count > 0 else 0.0
            print(f"  Selected {len(training_pairs)} diverse pairs (avg pairwise similarity: {avg_sim:.3f})")
        else:
            print(f"  Selected {len(training_pairs)} diverse pairs")
        
        return training_pairs