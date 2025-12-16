import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from tamp_improv.approaches.improvisational.heuristics.heuristic_crl_base import (
    CRLHeuristicBase,
    CRLHeuristicConfig,
    StateEncoder,
    ContrastiveReplayBuffer,
)
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


class CRLHeuristicDiscrete(CRLHeuristicBase):
    """CRL heuristic for discrete action spaces.

    Uses action enumeration: evaluates all possible actions and selects
    via softmax over negative distances.

    Encodes states only (not state-action pairs).
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        config: CRLHeuristicConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize discrete CRL heuristic.

        Args:
            training_data: Training data with shortcuts
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            system: TAMP system with env and perceiver
            config: Configuration for training
            seed: Random seed
        """
        super().__init__(training_data, graph_distances, system, config, seed)

    def _get_encoder_input_dim(self) -> int:
        """Get encoder input dimension (state_dim only for discrete)."""
        sample_state = self.state_node_pairs[0][0]
        state_flat = self._flatten_state(sample_state)
        return state_flat.shape[0]

    def _prepare_encoder_input(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare encoder input (just state for discrete).

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
            action: Action tensor (ignored for discrete)

        Returns:
            State tensor unchanged
        """
        return state

    def _uses_action_encoding(self) -> bool:
        """Discrete does not encode actions."""
        return False

    def _select_action_greedy(
        self,
        env: Any,
        current_state: "ObsType",
        goal_node: int,
    ) -> int | None:
        """Select action using greedy policy over learned embedding space.

        For discrete action spaces, we enumerate all actions, simulate each,
        compute distance in embedding space, and select via softmax.

        Args:
            env: Environment
            current_state: Current state
            goal_node: Goal node ID

        Returns:
            Selected action (int), or None if no valid actions
        """
        # Get available actions (discrete enumeration)
        available_actions = list(range(env.action_space.n))

        if len(available_actions) == 0:
            return None

        # If networks not trained yet, use random policy
        if self.s_encoder is None or self.g_encoder is None:
            print("Outputting random action")
            return random.choice(available_actions)

        # Get goal node embedding
        with torch.no_grad():
            goal_emb = self._encode_node(goal_node).cpu().numpy()  # (k,)

        # Compute distances in embedding space for each action
        distances = []
        for action in available_actions:
            # Save current state
            env.reset_from_state(current_state)

            # Take action to get next state
            next_state, _, _, _, _ = env.step(action)

            # Encode next state
            next_flat = self._flatten_state(next_state)
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_flat).unsqueeze(0).to(self.device)
                next_emb = self._encode_state(next_tensor).squeeze(0).cpu().numpy()  # (k,)

            # Compute L2 distance to goal in embedding space
            dist = 0.5 * np.linalg.norm(next_emb - goal_emb)**2
            distances.append(dist)

        distances = np.array(distances)

        # Greedy selection if temperature is 0
        if self.policy_temperature == 0:
            return available_actions[np.argmin(distances)]

        # Softmax over negative distances (lower distance = higher probability)
        logits = -distances / self.policy_temperature
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample action
        action = np.random.choice(available_actions, p=probabilities)
        return action

    def rollout(
        self,
        env: Any,
        start_state: "ObsType",
        goal_node: int,
        max_steps: int = 100,
        temperature: float | None = None,
    ) -> tuple[list[NDArray], list[int], bool, list[NDArray] | None]:
        """Rollout the learned policy from start state to goal node.

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
                - actions: None for discrete (we don't need to store actions)
        """
        if temperature is not None:
            old_temp = self.policy_temperature
            self.policy_temperature = temperature

        # Reset to start state
        current_state, _ = env.reset_from_state(start_state)

        states = [self._flatten_state(current_state)]
        nodes = [self.get_node(current_state)]

        success = False

        for _ in range(max_steps):
            # Check if reached goal
            current_node = self.get_node(current_state)
            if current_node == goal_node:
                success = True
                break

            # Select action
            action = self._select_action_greedy(env, current_state, goal_node)
            if action is None:
                break

            # Step environment
            current_state, _, _, _, _ = env.step(action)

            # Store state and node
            states.append(self._flatten_state(current_state))
            nodes.append(self.get_node(current_state))

        # Restore temperature
        if temperature is not None:
            self.policy_temperature = old_temp

        # Return None for actions (discrete doesn't need them for training)
        return states, nodes, success, None

    def train(
        self,
        state_node_pairs: list[tuple["ObsType", int]],
        num_epochs: int = 1000,
        trajectories_per_epoch: int = 10,
        max_episode_steps: int = 100,
    ) -> dict:
        """Train distance heuristic using contrastive state-node learning.

        Args:
            state_node_pairs: List of (source_state, target_node) pairs
            num_epochs: Number of training epochs
            trajectories_per_epoch: Number of trajectories to collect per epoch
            max_episode_steps: Maximum steps per episode

        Returns:
            Dictionary containing training history
        """
        print(f"\nTraining discrete CRL heuristic on {len(state_node_pairs)} state-node pairs...")
        print(f"Device: {self.device}")
        print(f"Number of nodes: {self.num_nodes}")

        # Determine state dimension
        state_dim = self._get_encoder_input_dim()
        print(f"State dimension: {state_dim}")
        print(f"Latent dimension: {self.config.latent_dim}")

        # Create state encoder network
        self.s_encoder = StateEncoder(
            input_dim=state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        # Initialize goal node embedding matrix (S x k)
        self.g_encoder = nn.Parameter(
            torch.randn(self.num_nodes, self.config.latent_dim, device=self.device) * 0.01
        )

        # Create optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            list(self.s_encoder.parameters()) + [self.g_encoder],
            lr=self.config.learning_rate
        )

        # Create cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )

        # Create replay buffer
        self.replay_buffer = ContrastiveReplayBuffer(
            capacity=self.config.buffer_size,
            gamma=self.config.gamma,
            repetition_factor=self.config.repetition_factor,
        )

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Collecting {trajectories_per_epoch} trajectories per epoch")

        # Temperature annealing parameters
        initial_temp = self.config.policy_temperature
        min_temp = self.config.eval_temperature

        # Initialize training history tracking
        training_history = {
            'total_loss': [],
            'alignment_loss': [],
            'uniformity_loss': [],
            'accuracy': [],
            'success_rate': [],
            'avg_success_length': [],
            'learning_rate': [],
            'policy_temperature': [],
        }

        # Training loop
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            # Update policy temperature using cosine annealing
            progress = (epoch+1) / num_epochs
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            self.policy_temperature = min_temp + (initial_temp - min_temp) * cosine_factor

            # Collect trajectories
            trajectories, trajectory_metadata, traj_stats = self.collect_trajectories(
                self.env,
                state_node_pairs,
                num_trajectories=trajectories_per_epoch,
                max_episode_steps=max_episode_steps,
            )

            # Store trajectories in replay buffer (no actions for discrete)
            for trajectory, metadata in zip(trajectories, trajectory_metadata):
                self.replay_buffer.store_trajectory(
                    states=trajectory['states'],
                    nodes=trajectory['nodes'],
                    actions=None,  # Discrete doesn't store actions
                    start_state=metadata['start_state'],
                    goal_node=metadata['goal_node']
                )
                self.total_episodes += 1

            # Train if we have enough data
            if (epoch > 0 and
                epoch % self.config.learn_frequency == 0 and
                len(self.replay_buffer) >= self.config.batch_size):

                metrics = self._train_step()

                # Track metrics in history
                current_lr = self.optimizer.param_groups[0]['lr']
                training_history['total_loss'].append(metrics['loss'])
                training_history['alignment_loss'].append(metrics['l_align'])
                training_history['uniformity_loss'].append(metrics['l_unif'])
                training_history['accuracy'].append(metrics['accuracy'])
                training_history['success_rate'].append(traj_stats['success_rate'])
                training_history['avg_success_length'].append(traj_stats['avg_success_length'])
                training_history['learning_rate'].append(current_lr)
                training_history['policy_temperature'].append(self.policy_temperature)

                # Logging
                print(f"\n[Epoch {epoch}/{num_epochs}]")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                print(f"  Learning rate: {current_lr:.6f}")
                print(f"  Policy temperature: {self.policy_temperature:.4f}")
                print(f"  --- Policy Performance ---")
                print(f"  Success rate: {traj_stats['success_rate']:.2%} ({traj_stats['num_successes']}/{traj_stats['num_trajectories']})")
                print(f"  Avg trajectory length: {traj_stats['avg_length']:.1f}")
                if traj_stats['num_successes'] > 0:
                    print(f"  Avg successful length: {traj_stats['avg_success_length']:.1f}")
                print(f"  --- Training Metrics ---")
                print(f"  Total loss: {metrics['loss']:.4f}")
                print(f"  Alignment loss: {metrics['l_align']:.4f}")
                print(f"  Uniformity loss: {metrics['l_unif']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")

            # Step the learning rate scheduler every epoch
            self.scheduler.step()

        print(f"\nTraining complete!")
        return training_history

    def latent_dist(self, source_state: "ObsType", target_node: int, action: np.ndarray | None = None) -> float:
        """Compute distance in latent space between state and node.

        Args:
            source_state: Source state
            target_node: Target node ID
            action: Action (ignored for discrete)

        Returns:
            L2 distance in embedding space
        """
        if self.s_encoder is None or self.g_encoder is None:
            return 0.0

        source_flat = self._flatten_state(source_state)

        # Encode using encoding functions
        with torch.no_grad():
            source_tensor = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            source_emb = self._encode_state(source_tensor).squeeze(0)
            target_emb = self._encode_node(target_node)

            # Compute L2 distance in embedding space
            distance = torch.norm(source_emb - target_emb).item()

        return float(distance)

    def estimate_distance(self, source_state: "ObsType", target_node: int) -> float:
        """Estimate trajectory distance from state to node.

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            Estimated number of steps to reach target from source
        """
        d_sg_sq = (self.latent_dist(source_state, target_node))**2
        d_gg_sq = 0
        target_states = self.node_to_states[target_node]

        n = 0
        for target_state in random.sample(target_states, min(100, len(target_states))):
            d_gg_sq += (self.latent_dist(target_state, target_node))**2
            n += 1
        d_gg_sq /= n

        print("D_gg:", d_gg_sq, "D_sg:", d_sg_sq)

        return (1 / (2 * np.log(self.config.gamma))) * (d_gg_sq - d_sg_sq)
