"""
SAC is the correct solution for picking actions, but we end up randomly sampling in heuristic_crl in
the continuous action space, since SAC has trouble learning distances (e.g., negative distances, policy
loss increasing)
"""
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
    contrastive_loss,
)
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


class PolicyNetwork(nn.Module):
    """Policy network that outputs actions given (state, goal_embedding).

    Takes concatenated (state, goal_embedding) and outputs mean and log_std for action distribution.

    Based on stablebaselines implementation for SAC
    """

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        """Initialize policy network.

        Args:
            state_dim: Dimension of state
            goal_dim: Dimension of goal embedding
            action_dim: Dimension of action
            hidden_dims: Hidden layer sizes
        """
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = state_dim + goal_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor, goal_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
            goal_emb: Goal embedding (batch_size, goal_dim) or (goal_dim,)

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        x = torch.cat([state, goal_emb], dim=-1)

        h = self.shared(x)

        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)

        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state: torch.Tensor, goal_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, goal_emb)
        std = torch.exp(log_std)

        distribution = torch.distributions.Normal(mean, std)
        gaussian_actions = distribution.rsample()
        actions = torch.tanh(gaussian_actions)

        log_prob = distribution.log_prob(gaussian_actions)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        log_prob -= torch.log(1 - actions.pow(2)).sum(dim=-1, keepdim=True)

        return actions, log_prob


class CRLHeuristicContinuous(CRLHeuristicBase):
    """CRL heuristic for continuous action spaces.

    Learns a policy network that outputs actions given (state, goal_node).

    Encodes (state, action) pairs.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        config: CRLHeuristicConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize continuous CRL heuristic.

        Args:
            training_data: Training data with shortcuts
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            system: TAMP system with env and perceiver
            config: Configuration for training
            seed: Random seed
        """
        super().__init__(training_data, graph_distances, system, config, seed)

        self.action_dim = system.env.action_space.shape[0]
        self.action_low = system.env.action_space.low
        self.action_high = system.env.action_space.high
        self.node_to_state_action_pairs: dict[int, list[tuple[NDArray, NDArray]]] = {}

        self.policy_network: PolicyNetwork | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None

    def _get_encoder_input_dim(self) -> int:
        """Get encoder input dimension (state_dim + action_dim for continuous)."""
        sample_state = self.state_node_pairs[0][0]
        state_flat = self._flatten_state(sample_state)
        return state_flat.shape[0] + self.action_dim

    def _prepare_encoder_input(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare encoder input (concatenate state and action for continuous).

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
            action: Action tensor (batch_size, action_dim) or (action_dim,)

        Returns:
            Concatenated (state, action) tensor
        """
        if action is None:
            raise ValueError("Continuous CRL requires action for encoding")

        if state.dim() == 1:
            return torch.cat([state, action], dim=0)
        else:
            return torch.cat([state, action], dim=1)

    def _uses_action_encoding(self) -> bool:
        """Continuous encodes (state, action) pairs."""
        return True

    def _select_action_greedy(
        self,
        env: Any,
        current_state: "ObsType",
        goal_node: int,
    ) -> np.ndarray | None:
        """Select action using learned policy network.

        Uses policy network to sample action given (state, goal_embedding).
        Falls back to random if policy not trained yet.

        Args:
            env: Environment
            current_state: Current state
            goal_node: Goal node ID

        Returns:
            Selected action, or None if error
        """
        if self.policy_network is None or self.g_encoder is None:
            print("Policy not trained, outputting random action")
            action = np.random.uniform(
                low=self.action_low,
                high=self.action_high,
                size=self.action_dim
            ).astype(np.float32)
            return action

        state_flat = self._flatten_state(current_state)
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            goal_emb = self._encode_node(goal_node).unsqueeze(0)  # (1, k)

            action_tensor, _ = self.policy_network.sample(state_tensor, goal_emb)
            action = action_tensor.squeeze(0).cpu().numpy()  # (action_dim,)

        # Scale from [-1, 1] (tanh output) to [action_low, action_high]
        action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        action = np.clip(action, self.action_low, self.action_high)

        return action.astype(np.float32)

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
                - actions: List of actions taken (for continuous)
        """
        if temperature is not None:
            old_temp = self.policy_temperature
            self.policy_temperature = temperature

        # Reset to start state
        current_state, _ = env.reset_from_state(start_state)

        states = [self._flatten_state(current_state)]

        # Try to get initial node - if state not in graph, can't proceed
        try:
            current_node = self.get_node(current_state)
            nodes = [current_node]
        except KeyError:
            # State not in planning graph - return empty trajectory
            return states, [goal_node], False, []

        actions = []

        success = False

        for _ in range(max_steps):
            if current_node == goal_node:
                success = True
                break

            action = self._select_action_greedy(env, current_state, goal_node)
            if action is None:
                break

            current_state, _, _, _, _ = env.step(action)

            try:
                current_node = self.get_node(current_state)
            except KeyError:
                # Reached state outside planning graph - terminate rollout
                break

            states.append(self._flatten_state(current_state))
            nodes.append(current_node)
            actions.append(action)

        if temperature is not None:
            self.policy_temperature = old_temp

        return states, nodes, success, actions

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
        print(f"\nTraining continuous CRL heuristic on {len(state_node_pairs)} state-node pairs...")
        print(f"Device: {self.device}")
        print(f"Number of nodes: {self.num_nodes}")

        encoder_input_dim = self._get_encoder_input_dim()
        print(f"Encoder input dimension (state + action): {encoder_input_dim}")
        print(f"Latent dimension: {self.config.latent_dim}")

        # Create state-action encoder network
        self.s_encoder = StateEncoder(
            input_dim=encoder_input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        # Initialize goal node embedding matrix (S x k)
        self.g_encoder = nn.Parameter(
            torch.randn(self.num_nodes, self.config.latent_dim, device=self.device) * 0.01
        )

        # Create policy network (state_dim, goal_dim, action_dim)
        sample_state = state_node_pairs[0][0]
        state_dim = self._flatten_state(sample_state).shape[0]
        self.policy_network = PolicyNetwork(
            state_dim=state_dim,
            goal_dim=self.config.latent_dim,
            action_dim=self.action_dim,
            hidden_dims=[256, 256],
        ).to(self.device)

        print(f"Policy network: state_dim={state_dim}, goal_dim={self.config.latent_dim}, action_dim={self.action_dim}")

        # Create optimizer for CRL parameters (encoder + goal embeddings)
        self.optimizer = torch.optim.Adam(
            list(self.s_encoder.parameters()) + [self.g_encoder],
            lr=self.config.learning_rate
        )

        # Create separate optimizer for policy network
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
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
            'policy_loss': [],
            'policy_distance': [],
            'policy_entropy': [],
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

            # Store trajectories in replay buffer (with actions for continuous)
            for trajectory, metadata in zip(trajectories, trajectory_metadata):
                self.replay_buffer.store_trajectory(
                    states=trajectory['states'],
                    nodes=trajectory['nodes'],
                    actions=trajectory['actions'],  # Store actions for continuous
                    start_state=metadata['start_state'],
                    goal_node=metadata['goal_node']
                )
                self.total_episodes += 1

                # Also store (state, action) pairs for each node
                # Note: actions[i] is the action taken FROM states[i] leading to states[i+1]
                states = trajectory['states']
                nodes = trajectory['nodes']
                actions = trajectory['actions']
                for i in range(len(actions)):
                    node_id = nodes[i]
                    state = states[i]
                    action = actions[i]
                    if node_id not in self.node_to_state_action_pairs:
                        self.node_to_state_action_pairs[node_id] = []
                    self.node_to_state_action_pairs[node_id].append((state, action))

            # Print trajectory stats every epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch}/{num_epochs}]")
            print(f"  Buffer size: {len(self.replay_buffer)}")
            print(f"  Learning rate: {current_lr:.6f}")
            print(f"  Policy temperature: {self.policy_temperature:.4f}")
            print(f"  --- Policy Performance ---")
            print(f"  Success rate: {traj_stats['success_rate']:.2%} ({traj_stats['num_successes']}/{traj_stats['num_trajectories']})")
            print(f"  Avg trajectory length: {traj_stats['avg_length']:.1f}")
            if traj_stats['num_successes'] > 0:
                print(f"  Avg successful length: {traj_stats['avg_success_length']:.1f}")

            # Train if we have enough data
            if (epoch > 0 and
                epoch % self.config.learn_frequency == 0 and
                len(self.replay_buffer) >= self.config.batch_size):

                metrics = self._train_step()

                # Track metrics in history
                training_history['total_loss'].append(metrics['loss'])
                training_history['alignment_loss'].append(metrics['l_align'])
                training_history['uniformity_loss'].append(metrics['l_unif'])
                training_history['accuracy'].append(metrics['accuracy'])
                training_history['success_rate'].append(traj_stats['success_rate'])
                training_history['avg_success_length'].append(traj_stats['avg_success_length'])
                training_history['learning_rate'].append(current_lr)
                training_history['policy_temperature'].append(self.policy_temperature)
                training_history['policy_loss'].append(metrics['policy_loss'])
                training_history['policy_distance'].append(metrics['policy_distance'])
                training_history['policy_entropy'].append(metrics['policy_entropy'])

                # Logging for training metrics
                print(f"  --- CRL Training Metrics ---")
                print(f"  Total loss: {metrics['loss']:.4f}")
                print(f"  Alignment loss: {metrics['l_align']:.4f}")
                print(f"  Uniformity loss: {metrics['l_unif']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  --- Policy Training Metrics ---")
                print(f"  Policy loss: {metrics['policy_loss']:.4f}")
                print(f"  Policy distance: {metrics['policy_distance']:.4f}")
                print(f"  Policy entropy: {metrics['policy_entropy']:.4f}")

            # Step the learning rate scheduler every epoch
            self.scheduler.step()

        print(f"\nTraining complete!")
        return training_history

    def _train_step(self) -> dict:
        """Perform one training step with joint CRL + Policy training.

        Trains:
        1. CRL embeddings using contrastive loss on trajectory data
        2. Policy network to output actions that minimize distance to goal

        Returns:
            Dictionary of training metrics
        """
        batch = self.replay_buffer.sample(
            self.config.batch_size,
            include_actions=True
        )

        current_states = batch["current_states"].to(self.device)
        current_actions = batch["current_actions"].to(self.device)
        future_nodes = batch["future_nodes"].to(self.device)

        current_inputs = self._prepare_encoder_input(current_states, current_actions)

        # Critic 

        for _ in range(self.config.iters_per_epoch):
            self.optimizer.zero_grad()

            loss, metrics = contrastive_loss(
                lambda x: self.s_encoder(x) if not self.config.normalize_embeddings
                         else F.normalize(self.s_encoder(x), p=2, dim=-1),
                self._encode_node,
                current_inputs,
                future_nodes,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.s_encoder.parameters()) + [self.g_encoder],
                max_norm=self.config.grad_clip
            )

            self.optimizer.step()

        # Actor

        self.policy_optimizer.zero_grad()

        goal_embeddings = torch.stack([self._encode_node(int(node)) for node in future_nodes])

        sampled_actions, log_probs = self.policy_network.sample(current_states, goal_embeddings)

        action_range = torch.FloatTensor(self.action_high - self.action_low).to(self.device)
        action_low_tensor = torch.FloatTensor(self.action_low).to(self.device)
        scaled_actions = action_low_tensor + (sampled_actions + 1.0) * 0.5 * action_range
        scaled_actions = torch.clamp(scaled_actions, action_low_tensor, action_low_tensor + action_range)

        policy_inputs = self._prepare_encoder_input(current_states, scaled_actions)
        policy_embeddings = self.s_encoder(policy_inputs)
        if self.config.normalize_embeddings:
            policy_embeddings = F.normalize(policy_embeddings, p=2, dim=-1)

        distances_sq = torch.sum((policy_embeddings - goal_embeddings) ** 2, dim=1, keepdim=True)

        alpha = 0.01 # JCZ: LOWERING SINCE ENTROPY COMPONENT BLEW UP
        policy_loss = (distances_sq + alpha * log_probs).mean()

        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=self.config.grad_clip
        )

        self.policy_optimizer.step()

        metrics['policy_loss'] = policy_loss.item()
        metrics['policy_distance'] = distances_sq.mean().item()
        metrics['policy_entropy'] = -log_probs.mean().item()

        return metrics

    def latent_dist(self, source_state: "ObsType", target_node: int, action: np.ndarray | None = None) -> float:
        """Compute distance in latent space between (state, action) and node.

        Args:
            source_state: Source state
            target_node: Target node ID
            action: Action (required for continuous)

        Returns:
            L2 distance in embedding space
        """
        if self.s_encoder is None or self.g_encoder is None:
            return 0.0

        if action is None:
            raise ValueError("Continuous CRL requires action for encoding")

        source_flat = self._flatten_state(source_state)

        # Encode (state, action) pair using encoding functions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

            # _encode_state calls _prepare_encoder_input which concatenates state and action
            source_emb = self._encode_state(state_tensor, action_tensor).squeeze(0)
            target_emb = self._encode_node(target_node)

            # Compute L2 distance in embedding space
            distance = torch.norm(source_emb - target_emb).item()

        return float(distance)

    def estimate_distance(self, source_state: "ObsType", target_node: int) -> float:
        """Estimate trajectory distance from state to node.

        For continuous action spaces, we use actions from collected trajectories.

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            Estimated number of steps to reach target from source
        """
        # Find the node for source_state
        source_node = self.get_node(source_state)

        assert source_node in self.node_to_state_action_pairs, \
            f"No trajectory data for source node {source_node}"
        assert len(self.node_to_state_action_pairs[source_node]) > 0, \
            f"Empty trajectory data for source node {source_node}"

        source_flat = self._flatten_state(source_state)
        matching_pairs = [(s, a) for s, a in self.node_to_state_action_pairs[source_node]
                        if np.allclose(s, source_flat)]
        if matching_pairs:
            _, source_action = matching_pairs[0]
        else:
            assert False, f"State not found in trajectory data for node {source_node}"

        d_sg_sq = (self.latent_dist(source_state, target_node, source_action))**2

        # For target node, use (state, action) pairs from trajectories
        assert target_node in self.node_to_state_action_pairs, \
            f"No trajectory data for target node {target_node}"
        assert len(self.node_to_state_action_pairs[target_node]) > 0, \
            f"Empty trajectory data for target node {target_node}"

        d_gg_sq = 0
        pairs = self.node_to_state_action_pairs[target_node]
        n = 0
        for state, action in random.sample(pairs, min(100, len(pairs))):
            d_gg_sq += (self.latent_dist(state, target_node, action))**2
            n += 1
        d_gg_sq /= n

        print("D_gg:", d_gg_sq, "D_sg:", d_sg_sq)

        return (1 / (2 * np.log(self.config.gamma))) * (d_gg_sq - d_sg_sq)
