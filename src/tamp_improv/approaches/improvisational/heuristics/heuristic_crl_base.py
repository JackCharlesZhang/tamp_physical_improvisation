"""Base class for CRL heuristics supporting both discrete and continuous action spaces.

This module contains the shared logic for contrastive state-node learning that is
independent of action space type. Subclasses implement action selection for discrete
or continuous action spaces.

Key difference:
- Discrete: Encodes states only, simulates actions during selection
- Continuous: Encodes (state, action) pairs, no simulation during selection
"""

import time
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class CRLHeuristicConfig:
    """Configuration for contrastive state-node distance heuristic."""

    # Network architecture
    latent_dim: int = 32  # Dimension of embedding space (k)
    hidden_dims: list[int] | None = None  # Hidden layer sizes [64, 64]
    normalize_embeddings: bool = True  # Whether to L2-normalize embeddings to unit sphere

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    buffer_size: int = 10000
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Contrastive learning
    gamma: float = 0.99  # For geometric sampling of future states
    repetition_factor: int = 4  # CRTR: repeat each trajectory this many times in batch
    policy_temperature: float = 1.0
    eval_temperature: float = 0.0

    # Training
    iters_per_epoch: int = 1  # Gradient steps per training call
    learn_frequency: int = 10  # Learn every N epochs
    num_rounds: int = 5
    num_epochs_per_round: int = 200
    trajectories_per_epoch: int = 10
    max_episode_steps: int = 100
    keep_fraction: float = 0.5

    # Multi-round training
    num_rounds: int = 1  # Number of pruning rounds (1 = no pruning, just normal train)
    keep_fraction: float = 0.1  # Fraction of node-node pairs to keep each round
    exploration_factor: float = 0

    # SAC pre-training parameters (continuous action spaces only)
    sac_num_episodes: int = 1000  # Number of SAC training episodes
    sac_max_steps: int = 100  # Max steps per SAC episode
    sac_batch_size: int = 256  # Batch size for SAC updates
    sac_buffer_capacity: int = 100000  # SAC replay buffer capacity
    sac_hidden_dim: int = 256  # Hidden layer size for SAC networks
    sac_lr: float = 3e-4  # Learning rate for SAC
    sac_gamma: float = 0.99  # Discount factor for SAC
    sac_tau: float = 0.005  # Soft update coefficient for target networks
    sac_alpha: float = 0.2  # Initial entropy coefficient
    sac_auto_alpha: bool = True  # Automatic entropy tuning

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


class StateEncoder(nn.Module):
    """Neural network that encodes states to k-dimensional embeddings.

    For discrete action spaces: Maps state -> embedding
    For continuous action spaces: Maps (state, action) -> embedding
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int] | None = None):
        """Initialize the encoder network.

        Args:
            input_dim: Dimension of input (state_dim for discrete, state_dim+action_dim for continuous)
            latent_dim: Dimension of output embedding (k)
            hidden_dims: List of hidden layer sizes
        """
        super(StateEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Default hidden dims
        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs to embeddings.

        Args:
            inputs: Batch of inputs (batch_size, input_dim) - float tensor
                   For discrete: just states
                   For continuous: concatenated (state, action)

        Returns:
            Embeddings (batch_size, latent_dim)
        """
        return self.network(inputs)


class ContrastiveReplayBuffer:
    """Replay buffer for storing complete trajectories.

    Implements CRTR (Contrastive Representations for Temporal Reasoning) sampling
    to encourage learning temporal dynamics rather than static context.

    Stores trajectories as sequences of (state, action, node) tuples.
    """

    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        repetition_factor: int = 4,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of trajectories to store
            gamma: Discount factor (used for geometric sampling of future states)
            repetition_factor: Number of times to repeat each trajectory in a batch
                              (for CRTR within-trajectory negatives)
        """
        self.capacity = capacity
        self.gamma = gamma
        self.repetition_factor = repetition_factor

        # Storage for complete trajectories
        # Each trajectory is a dict with:
        #   - 'states': list of flattened states (NDArray)
        #   - 'actions': list of actions (NDArray) - for continuous action spaces
        #   - 'nodes': list of node IDs (int)
        self.trajectories: deque = deque(maxlen=capacity)

        # Store metadata for each trajectory (start state, goal node)
        self.trajectory_metadata: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return total number of trajectories stored."""
        return len(self.trajectories)

    def store_trajectory(
        self,
        states: list[NDArray],
        nodes: list[int],
        actions: list[NDArray] | None = None,
        start_state: NDArray | None = None,
        goal_node: int | None = None
    ) -> None:
        """Store complete trajectory with metadata.

        Args:
            states: List of flattened states visited during episode
            nodes: List of node IDs corresponding to states
            actions: List of actions taken (optional, for continuous action spaces)
            start_state: Starting state of the trajectory
            goal_node: Goal node ID for the trajectory
        """
        if len(states) < 2 or len(states) != len(nodes):
            return

        # Store the trajectory
        traj_dict = {
            'states': states,
            'nodes': nodes,
        }
        if actions is not None:
            traj_dict['actions'] = actions

        self.trajectories.append(traj_dict)

        # Store metadata
        self.trajectory_metadata.append({
            'start_state': start_state,
            'goal_node': goal_node,
        })

    def sample(self, batch_size: int, include_actions: bool = False) -> dict[str, torch.Tensor]:
        """Sample batch using CRTR strategy.

        Implements Algorithm 1 from "Contrastive Representations for Temporal Reasoning":
        For each sample in batch:
        1. Sample a trajectory uniformly
        2. Sample t0 uniformly from [0, T-1]
        3. Sample t1 from geometric distribution starting at t0+1

        Args:
            batch_size: Number of samples to return
            include_actions: Whether to include actions in returned batch

        Returns:
            Dictionary containing:
                - current_states: (batch_size, state_dim) - states at time t0
                - current_actions: (batch_size, action_dim) - actions at time t0 (if include_actions=True)
                - future_nodes: (batch_size,) - node IDs at time t1
        """
        if len(self.trajectories) == 0:
            raise ValueError("Buffer is empty")

        # Ensure batch_size is divisible by repetition_factor
        assert batch_size % self.repetition_factor == 0

        # Sample unique trajectory IDs
        num_unique_trajs = batch_size // self.repetition_factor
        unique_traj_ids = np.random.choice(
            len(self.trajectories),
            size=num_unique_trajs,
            replace=False if num_unique_trajs <= len(self.trajectories) else True
        )

        # Repeat each trajectory ID repetition_factor times
        traj_ids = np.repeat(unique_traj_ids, self.repetition_factor)

        current_states = []
        current_actions = []
        future_nodes = []

        # For each trajectory in the batch, sample (t0, t1) pair
        for traj_id in traj_ids:
            trajectory = self.trajectories[traj_id]
            states = trajectory['states']
            nodes = trajectory['nodes']
            actions = trajectory.get('actions', None)
            traj_len = len(states)

            if traj_len < 2:
                current_states.append(states[0])
                if include_actions and actions is not None:
                    current_actions.append(actions[0])
                future_nodes.append(nodes[-1])
                continue

            # Sample t0 uniformly from trajectory
            t0 = np.random.randint(0, traj_len - 1)

            # Sample t1 using geometric distribution
            num_future = traj_len - t0 - 1
            p = 1 - self.gamma
            offset = min(
                np.random.geometric(p) - 1,
                num_future - 1
            )
            t1 = t0 + 1 + offset

            current_states.append(states[t0])
            if include_actions and actions is not None:
                current_actions.append(actions[t0])
            future_nodes.append(nodes[t1])

        # Stack into tensors
        result = {
            "current_states": torch.FloatTensor(np.array(current_states)),
            "future_nodes": torch.LongTensor(future_nodes),
        }

        if include_actions and current_actions:
            result["current_actions"] = torch.FloatTensor(np.array(current_actions))

        return result


def contrastive_loss(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    encode_node_fn: Callable[[torch.Tensor], torch.Tensor],
    current_inputs: torch.Tensor,
    future_nodes: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Compute contrastive loss with embeddings.

    This implements a simplified version of the contrastive loss:
    - phi = encode_fn(current_input)  # input = state for discrete, (state,action) for continuous
    - psi = encode_node_fn(future_node)
    - l_align: ||phi - psi||^2 (align positive pairs)
    - l_unif: logsumexp over negative pair distances (uniformity)

    Args:
        encode_fn: Function to encode inputs to embeddings
        encode_node_fn: Function to encode node IDs to embeddings
        current_inputs: Batch of current inputs (batch_size, input_dim)
        future_nodes: Batch of future node IDs (batch_size,)

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Get embeddings using provided encoding functions
    phi = encode_fn(current_inputs)  # (batch_size, k)
    psi = encode_node_fn(future_nodes)  # (batch_size, k)

    batch_size = phi.shape[0]

    # Alignment loss: ||phi - psi||^2 for positive pairs
    l_align = torch.mean((phi - psi) ** 2, dim=1)  # (batch_size,)

    # Pairwise distances: ||phi[i] - psi[j]||^2 for all i, j
    pdist = torch.sum((phi[:, None] - psi[None]) ** 2, dim=-1)  # (batch_size, batch_size)

    # Uniformity loss: logsumexp over negative pairs
    # Mask out diagonal (positive pairs) with identity matrix
    I = torch.eye(batch_size, device=phi.device)

    l_unif = (
        torch.logsumexp(-(pdist * (1 - I)), dim=1) +
        torch.logsumexp(-(pdist.T * (1 - I)), dim=1)
    ) / 2.0

    # Combined contrastive loss
    loss = l_align + l_unif

    # Accuracy: how often is the closest future node the correct one?
    closest_node_indices = torch.argmin(pdist, dim=1)
    predicted_nodes = future_nodes[closest_node_indices]
    true_nodes = future_nodes
    accuracy = torch.mean((predicted_nodes == true_nodes).float())

    # Total loss
    total_loss = loss.mean()

    # Metrics
    metrics = {
        'loss': total_loss.item(),
        'l_unif': l_unif.mean().item(),
        'l_align': l_align.mean().item(),
        'accuracy': accuracy.item(),
    }

    return total_loss, metrics


class CRLHeuristicBase(BaseHeuristic, ABC):
    """Base class for CRL heuristic with shared training logic.

    Subclasses must implement:
    - _select_action_greedy() for their specific action space
    - _get_encoder_input_dim() to specify encoder input size
    - _prepare_encoder_input() to format inputs for the encoder
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        config: CRLHeuristicConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize distance heuristic.

        Args:
            training_data: Training data with shortcuts
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            system: TAMP system with env and perceiver
            config: Configuration for training
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        self.env = system.env
        self.perceiver = system.perceiver
        self.node_to_states = training_data.node_states
        self.graph_distances = graph_distances
        self.training_data = training_data

        # Extract state-node pairs (state from node A, target node B)
        self.atoms_to_node = {}
        planning_graph = training_data.graph
        for node in planning_graph.nodes:
            self.atoms_to_node[node.atoms] = node.id

        state_node_pairs = []
        for node_id, states in self.node_to_states.items():
            for state in states:
                # Add pairs to all other nodes
                for target_node in planning_graph.nodes:
                    if target_node.id != node_id:
                        state_node_pairs.append((state, target_node.id))

        self.state_node_pairs = state_node_pairs

        # Create get_node function
        def get_node(state: ObsType) -> int:
            """Convert state to node ID."""
            atoms = self.perceiver.step(state)
            return self.atoms_to_node[frozenset(atoms)]

        self.get_node = get_node
        self.num_nodes = len(self.atoms_to_node)

        if self.config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Networks
        self.s_encoder: StateEncoder | None = None  # State encoder network (or state-action for continuous)
        self.g_encoder: torch.Tensor | None = None  # Goal node embedding matrix (S x k)
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ContrastiveReplayBuffer | None = None

        # Training state
        self.total_episodes = 0
        self.policy_temperature = config.policy_temperature

    @abstractmethod
    def _get_encoder_input_dim(self) -> int:
        """Get the input dimension for the encoder.

        Returns:
            Input dimension (state_dim for discrete, state_dim+action_dim for continuous)
        """
        pass

    @abstractmethod
    def _prepare_encoder_input(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare input for the encoder.

        Args:
            state: State tensor
            action: Action tensor (optional, for continuous action spaces)

        Returns:
            Encoder input tensor
        """
        pass

    @abstractmethod
    def _uses_action_encoding(self) -> bool:
        """Whether this heuristic encodes (state, action) pairs.

        Returns:
            True for continuous (encodes state-action), False for discrete (encodes state only)
        """
        pass

    def _encode_state(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """Encode state (and optionally action) to embedding.

        Args:
            state_tensor: State tensor (batch_size, state_dim) or (state_dim,)
            action_tensor: Action tensor (batch_size, action_dim) or (action_dim,) - optional

        Returns:
            Embedding tensor with same batch dimensions
        """
        encoder_input = self._prepare_encoder_input(state_tensor, action_tensor)
        emb = self.s_encoder(encoder_input)
        if self.config.normalize_embeddings:
            # Normalize to unit L2 norm
            return F.normalize(emb, p=2, dim=-1)
        return emb

    def _encode_node(self, node_id: int | torch.Tensor) -> torch.Tensor:
        """Encode node ID(s) to embedding.

        Args:
            node_id: Node ID (int) or tensor of node IDs (batch_size,)

        Returns:
            Embedding tensor (latent_dim,) or (batch_size, latent_dim)
        """
        emb = self.g_encoder[node_id]
        if self.config.normalize_embeddings:
            # Normalize to unit L2 norm
            return F.normalize(emb, p=2, dim=-1)
        return emb

    @abstractmethod
    def _select_action_greedy(
        self,
        env: Any,
        current_state: ObsType,
        goal_node: int,
    ) -> int | np.ndarray | None:
        """Select action using greedy policy over learned embedding space.

        Must be implemented by subclasses for discrete or continuous action spaces.

        Args:
            env: Environment
            current_state: Current state
            goal_node: Goal node ID

        Returns:
            Selected action (int for discrete, np.ndarray for continuous), or None if no valid actions
        """
        pass

    @abstractmethod
    def rollout(
        self,
        env: Any,
        start_state: ObsType,
        goal_node: int,
        max_steps: int = 100,
        temperature: float | None = None,
    ) -> tuple[list[NDArray], list[int], bool, list[NDArray] | None]:
        """Rollout the learned policy from start state to goal node.

        Must be implemented by subclasses to handle action collection differently.

        Args:
            env: Environment
            start_state: Starting state
            goal_node: Goal node ID to reach
            max_steps: Maximum number of steps
            temperature: Policy temperature (uses self.policy_temperature if None)

        Returns:
            Tuple of (states, nodes, success, actions) where:
                - states: List of flattened states visited
                - nodes: List of node IDs visited
                - success: Whether the goal node was reached
                - actions: List of actions taken (None for discrete, needed for continuous training)
        """
        pass

    def collect_trajectories(
        self,
        env: Any,
        state_node_pairs: list[tuple[ObsType, int]],
        num_trajectories: int,
        max_episode_steps: int = 100,
    ) -> tuple[list[dict], list[dict], dict]:
        """Collect trajectories by rolling out from state-node pairs.

        Uses learned policy (greedy over embedding space) if available,
        otherwise uses random policy.

        Args:
            env: Base environment (must support reset_from_state)
            state_node_pairs: List of (source_state, target_node) pairs
            num_trajectories: Number of trajectories to collect
            max_episode_steps: Maximum steps per trajectory

        Returns:
            Tuple of (trajectories, trajectory_metadata, stats_dict)
        """
        trajectories = []
        trajectory_metadata = []
        successes = []
        lengths = []

        for _ in range(num_trajectories):
            # Sample random state-node pair
            source_state, target_node = random.choice(state_node_pairs)

            # Rollout from source to target
            states, nodes, success, actions = self.rollout(
                env, source_state, target_node, max_steps=max_episode_steps
            )

            traj_dict = {
                'states': states,
                'nodes': nodes,
            }
            if actions is not None:
                traj_dict['actions'] = actions

            trajectories.append(traj_dict)
            trajectory_metadata.append({
                'start_state': self._flatten_state(source_state),
                'goal_node': target_node,
            })
            successes.append(success)
            lengths.append(len(states))

        # Compute statistics
        success_rate = np.mean(successes) if successes else 0.0
        avg_length = np.mean(lengths) if lengths else 0.0
        success_lengths = [l for l, s in zip(lengths, successes) if s]
        avg_success_length = np.mean(success_lengths) if success_lengths else 0.0

        stats = {
            'success_rate': success_rate,
            'avg_length': avg_length,
            'avg_success_length': avg_success_length,
            'num_successes': sum(successes),
            'num_trajectories': num_trajectories
        }

        return trajectories, trajectory_metadata, stats

    def multi_train(self) -> dict:
        """Multi-round training that iteratively prunes to promising shortcuts.

        Returns:
            Dictionary with combined training history across all rounds
        """
        state_node_pairs = self.state_node_pairs
        graph_distances = self.graph_distances

        num_rounds = self.config.num_rounds
        num_epochs_per_round = self.config.num_epochs_per_round
        trajectories_per_epoch = self.config.trajectories_per_epoch
        max_episode_steps = self.config.max_episode_steps
        keep_fraction = self.config.keep_fraction

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING: {num_rounds} rounds, {num_epochs_per_round} epochs/round")
        print(f"Starting with {len(state_node_pairs)} state-node pairs")
        print(f"Keep fraction: {keep_fraction} (pruning {1-keep_fraction:.1%} each round)")
        print(f"{'='*80}\n")

        # Build mapping: (source_node, target_node) -> list of (source_state, target_node) pairs
        print("Building node-node to state-node mapping...")
        node_pair_to_state_pairs: dict[tuple[int, int], list[tuple[ObsType, int]]] = {}
        for state, target_node in state_node_pairs:
            source_node = self.get_node(state)
            key = (source_node, target_node)
            if key not in node_pair_to_state_pairs:
                node_pair_to_state_pairs[key] = []
            node_pair_to_state_pairs[key].append((state, target_node))

        print(f"Found {len(node_pair_to_state_pairs)} unique node-node pairs")
        print(f"Average {len(state_node_pairs) / len(node_pair_to_state_pairs):.1f} state-node pairs per node-node pair")

        # Combined history across all rounds
        combined_history = {
            'total_loss': [],
            'alignment_loss': [],
            'uniformity_loss': [],
            'accuracy': [],
            'success_rate': [],
            'avg_success_length': [],
            'learning_rate': [],
            'policy_temperature': [],
            'round_boundaries': [],
            'num_state_pairs_per_round': [],
            'num_node_pairs_per_round': [],
            'distance_matrices': [],
            'graph_distances': graph_distances,
        }

        # Start with all node-node pairs
        current_node_pairs = list(node_pair_to_state_pairs.keys())

        for round_idx in range(num_rounds):
            # Get all state-node pairs for current node-node pairs
            current_state_pairs = []
            for node_pair in current_node_pairs:
                current_state_pairs.extend(node_pair_to_state_pairs[node_pair])

            print(f"\n{'='*80}")
            print(f"ROUND {round_idx + 1}/{num_rounds}")
            print(f"Training on {len(current_node_pairs)} node-node pairs")
            print(f"  -> {len(current_state_pairs)} state-node pairs")
            print(f"{'='*80}\n")

            # Mark start of this round in history
            combined_history['round_boundaries'].append(len(combined_history['total_loss']))
            combined_history['num_state_pairs_per_round'].append(len(current_state_pairs))
            combined_history['num_node_pairs_per_round'].append(len(current_node_pairs))

            # Train on current subset
            round_history = self.train(
                state_node_pairs=current_state_pairs,
                num_epochs=num_epochs_per_round,
                trajectories_per_epoch=trajectories_per_epoch,
                max_episode_steps=self.config.max_episode_steps,
            )

            # Append this round's history
            for key in ['total_loss', 'alignment_loss', 'uniformity_loss', 'accuracy',
                       'success_rate', 'avg_success_length', 'learning_rate', 'policy_temperature']:
                combined_history[key].extend(round_history[key])

            # Compute distance matrix
            print(f"\n[DEBUG] Computing distance matrix for ALL {len(node_pair_to_state_pairs)} node pairs after round {round_idx + 1}...")
            all_node_pairs = list(node_pair_to_state_pairs.keys())
            distance_matrix = self._compute_distance_matrix(all_node_pairs)
            combined_history['distance_matrices'].append({
                'round': round_idx + 1,
                'distances': distance_matrix,
                'active_node_pairs': current_node_pairs.copy(),
            })
            print(f"[DEBUG] Distance matrix computed with {len(distance_matrix)} entries")

            # Pruning
            if round_idx < num_rounds - 1:
                print(f"\n{'='*80}")
                print(f"PRUNING after round {round_idx + 1}")
                print(f"{'='*80}\n")

                new_data = self.prune(add_exploration=True)
                selected_node_pairs_set = new_data.unique_shortcuts
                current_node_pairs = list(selected_node_pairs_set)

                resulting_state_pairs = sum(
                    len(node_pair_to_state_pairs.get((src, tgt), []))
                    for src, tgt in current_node_pairs
                )

                print(f"Pruned node-node pairs: {len(all_node_pairs)} -> {len(current_node_pairs)}")
                print(f"Resulting state-node pairs: {resulting_state_pairs}")

        # Final state-node pairs
        final_state_pairs = []
        for node_pair in current_node_pairs:
            final_state_pairs.extend(node_pair_to_state_pairs[node_pair])

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING COMPLETE")
        print(f"Final node-node pairs: {len(current_node_pairs)}")
        print(f"Final state-node pairs: {len(final_state_pairs)}")
        print(f"{'='*80}\n")

        return combined_history

    def _compute_distance_matrix(self, node_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Compute estimated distances for all node pairs."""
        distance_matrix = {}
        for source_node, target_node in node_pairs:
            est_dist = self.estimate_node_distance(source_node, target_node)
            distance_matrix[(source_node, target_node)] = est_dist
        return distance_matrix

    @abstractmethod
    def train(
        self,
        state_node_pairs: list[tuple[ObsType, int]],
        num_epochs: int = 1000,
        trajectories_per_epoch: int = 10,
        max_episode_steps: int = 100,
    ) -> dict:
        """Train distance heuristic.

        Must be implemented by subclasses due to differences in encoder initialization
        and training loop.
        """
        pass

    def _train_step(self) -> dict:
        """Perform one training step (batch update).

        Returns:
            Dictionary of training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(
            self.config.batch_size,
            include_actions=self._uses_action_encoding()
        )

        # Move to device
        current_states = batch["current_states"].to(self.device)
        future_nodes = batch["future_nodes"].to(self.device)

        # Prepare encoder inputs
        if self._uses_action_encoding():
            current_actions = batch["current_actions"].to(self.device)
            current_inputs = self._prepare_encoder_input(current_states, current_actions)
        else:
            current_inputs = self._prepare_encoder_input(current_states)

        # Run gradient descent
        for _ in range(self.config.iters_per_epoch):
            self.optimizer.zero_grad()

            # Compute loss using encoding functions
            loss, metrics = contrastive_loss(
                lambda x: self.s_encoder(x) if not self.config.normalize_embeddings
                         else F.normalize(self.s_encoder(x), p=2, dim=-1),
                self._encode_node,
                current_inputs,
                future_nodes,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.s_encoder.parameters()) + [self.g_encoder],
                max_norm=self.config.grad_clip
            )

            # Update parameters
            self.optimizer.step()

        return metrics

    @abstractmethod
    def latent_dist(self, source_state: ObsType, target_node: int, action: np.ndarray | None = None) -> float:
        """Compute distance in latent space.

        Args:
            source_state: Source state
            target_node: Target node ID
            action: Action (for continuous case)

        Returns:
            L2 distance in embedding space
        """
        pass

    @abstractmethod
    def estimate_distance(self, source_state: ObsType, target_node: int) -> float:
        """Estimate trajectory distance from state to node."""
        pass

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Estimate distance between two nodes."""
        source_states = self.node_to_states[source_node]
        avg = 0
        n = 0
        for source_state in random.sample(source_states, min(100, len(source_states))):
            avg += self.estimate_distance(source_state, target_node)
            n += 1
        return avg / n

    def prune(
        self, add_exploration: bool = False, **kwargs: Any
    ) -> GoalConditionedTrainingData:
        """Prune node pairs based on estimated vs graph distances."""
        print(f"\n[DEBUG] Pruning with keep_fraction={self.config.keep_fraction}, max_episode_steps={self.config.max_episode_steps}")

        node_pair_scores = []
        filtered_by_max_steps = 0

        for (source_node, target_node), graph_dist in self.graph_distances.items():
            est_dist = self.estimate_node_distance(source_node, target_node)

            if self.config.max_episode_steps is not None and est_dist > self.config.max_episode_steps:
                filtered_by_max_steps += 1
                continue

            score = est_dist - graph_dist
            node_pair_scores.append((score, source_node, target_node, est_dist))

        print(f"[DEBUG] Scored {len(node_pair_scores)} node pairs")
        if self.config.max_episode_steps is not None:
            print(f"[DEBUG] Filtered out {filtered_by_max_steps} pairs with est_dist > {self.config.max_episode_steps}")

        node_pair_scores.sort(key=lambda x: x[0])

        shortcuts = set()
        for score, source_node, target_node, est_dist in node_pair_scores:
            if score < 0:
                shortcuts.add((source_node, target_node))

        print(f"[DEBUG] Found {len(shortcuts)} shortcuts (negative scores)")

        num_to_keep = max(1, int(len(node_pair_scores) * self.config.keep_fraction))
        top_pairs = set()
        for score, source_node, target_node, est_dist in node_pair_scores[:num_to_keep]:
            top_pairs.add((source_node, target_node))

        print(f"[DEBUG] Keeping top {num_to_keep} pairs ({self.config.keep_fraction:.1%})")

        if add_exploration:
            num_exploration = int(len(self.graph_distances) * self.config.exploration_factor)
            all_pairs = set(self.graph_distances.keys())
            unexplored_pairs = all_pairs - shortcuts - top_pairs
            exploration_pairs = random.sample(unexplored_pairs, min(num_exploration, len(unexplored_pairs)))
            for source_node, target_node in exploration_pairs:
                top_pairs.add((source_node, target_node))
            print(f"[DEBUG] Added {len(exploration_pairs)} exploration pairs")

        result = shortcuts | top_pairs
        print(f"[DEBUG] Total pairs to keep: {len(result)} (union of shortcuts and top)")

        selected_set = set(result)
        selected_indices = []
        for i, (source_id, target_id) in enumerate(self.training_data.valid_shortcuts):
            if (source_id, target_id) in selected_set:
                selected_indices.append(i)

        print(f"  ({len(selected_indices)} state-node pairs)")

        original_shortcut_info = self.training_data.config.get("shortcut_info", [])
        pruned_shortcut_info = (
            [original_shortcut_info[i] for i in selected_indices]
            if original_shortcut_info
            else []
        )

        pruned_data = GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[self.training_data.current_atoms[i] for i in selected_indices],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[self.training_data.valid_shortcuts[i] for i in selected_indices],
            unique_shortcuts=list(selected_set),
            node_states=self.training_data.node_states,
            node_atoms=self.training_data.node_atoms,
            graph=self.training_data.graph,
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "crl",
            },
        )

        return pruned_data

    def _flatten_state(self, state: ObsType) -> np.ndarray:
        """Flatten state to array."""
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32)
        return np.array(state).flatten().astype(np.float32)

    def save(self, path: str) -> None:
        """Save distance heuristic to disk."""
        os.makedirs(path, exist_ok=True)

        torch.save(self.s_encoder.state_dict(), f"{path}/s_encoder.pt")
        torch.save(self.g_encoder, f"{path}/g_encoder.pt")

        with open(f"{path}/config.pkl", "wb") as f:
            pickle.dump(self.config, f)

        print(f"Distance heuristic saved to {path}")

    def load(self, path: str, input_dim: int, num_nodes: int) -> None:
        """Load distance heuristic from disk."""
        with open(f"{path}/config.pkl", "rb") as f:
            self.config = pickle.load(f)

        self.num_nodes = num_nodes

        self.s_encoder = StateEncoder(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        self.s_encoder.load_state_dict(torch.load(f"{path}/s_encoder.pt", map_location=self.device))
        self.s_encoder.eval()

        self.g_encoder = torch.load(f"{path}/g_encoder.pt", map_location=self.device)

        print(f"Distance heuristic loaded from {path}")
