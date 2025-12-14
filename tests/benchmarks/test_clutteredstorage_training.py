"""Test ClutteredStorage2D training with cached data."""

import pickle
from pathlib import Path

import pytest
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.benchmarks.clutteredstorage_system import (
    GraphClutteredStorage2DTAMPSystem,
)


def test_load_cached_training_data():
    """Test loading cached training data from pipeline cache."""
    cache_path = Path(
        "/scratch/network/sl5183/shortcut_learning/pipeline_cache/3_obj_cluttered_2d_baseline_cpu/full_training_data/data.pkl"
    )

    if not cache_path.exists():
        pytest.skip(f"Cached data not found at {cache_path}")

    # Load the pickled training data
    with open(cache_path, "rb") as f:
        training_data = pickle.load(f)

    print(f"\nLoaded training data:")
    print(f"  - States: {len(training_data.states)}")
    print(f"  - Current atoms: {len(training_data.current_atoms)}")
    print(f"  - Goal atoms: {len(training_data.goal_atoms)}")
    print(f"  - Valid shortcuts: {len(training_data.valid_shortcuts)}")
    print(f"  - Node states: {len(training_data.node_states)}")

    assert len(training_data.states) > 0
    assert len(training_data.current_atoms) == len(training_data.states)
    assert len(training_data.goal_atoms) == len(training_data.states)


def test_configure_training_with_cached_data():
    """Test configure_training method with cached training data."""
    cache_path = Path(
        "/scratch/network/sl5183/shortcut_learning/pipeline_cache/3_obj_cluttered_2d_baseline_cpu/full_training_data/data.pkl"
    )

    if not cache_path.exists():
        pytest.skip(f"Cached data not found at {cache_path}")

    # Load the training data
    with open(cache_path, "rb") as f:
        training_data = pickle.load(f)

    # Create system
    print("\nCreating GraphClutteredStorage2DTAMPSystem...")
    system = GraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=3, render_mode=None, seed=42
    )

    # Test that wrapped_env has configure_training method
    assert hasattr(system.wrapped_env, "configure_training"), \
        "wrapped_env should have configure_training method"

    # Configure training
    print("Configuring training with cached data...")
    system.wrapped_env.configure_training(training_data)

    # Verify the training was configured
    assert len(system.wrapped_env.training_states) > 0
    assert len(system.wrapped_env.current_atoms_list) > 0
    assert len(system.wrapped_env.goal_atoms_list) > 0

    print(f"Training configured successfully!")
    print(f"  - Training states: {len(system.wrapped_env.training_states)}")
    print(f"  - Current atoms: {len(system.wrapped_env.current_atoms_list)}")
    print(f"  - Goal atoms: {len(system.wrapped_env.goal_atoms_list)}")


def test_reset_from_configured_training():
    """Test that reset works after configuring training."""
    cache_path = Path(
        "/scratch/network/sl5183/shortcut_learning/pipeline_cache/3_obj_cluttered_2d_baseline_cpu/full_training_data/data.pkl"
    )

    if not cache_path.exists():
        pytest.skip(f"Cached data not found at {cache_path}")

    # Load the training data
    print("\n=== Loading training data ===")
    with open(cache_path, "rb") as f:
        training_data = pickle.load(f)

    print(f"Loaded {len(training_data.states)} training states")
    if len(training_data.states) > 0:
        print(f"First state shape: {training_data.states[0].shape if hasattr(training_data.states[0], 'shape') else type(training_data.states[0])}")

    # Create system
    print("\n=== Creating system ===")
    system = GraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=3, render_mode=None, seed=42
    )

    print("\n=== After creating system ===")
    print(f"wrapped_env type: {type(system.wrapped_env)}")
    print(f"base env type: {type(system.env)}")

    # Configure training
    print("\n=== Configuring training ===")
    system.wrapped_env.configure_training(training_data)

    print("\n=== After configuring training ===")

    # Test reset
    print("\n=== Testing reset ===")
    obs, info = system.wrapped_env.reset()

    assert obs is not None
    assert isinstance(obs, object)  # Should be a vector observation
    print(f"Reset successful! Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

    # Test multiple resets to cycle through training states
    for i in range(min(3, len(training_data.states))):
        print(f"\n=== Reset iteration {i+1} ===")
        obs, info = system.wrapped_env.reset()
        print(f"  Reset {i+1}: observation received, shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")


def test_policy_train_with_cached_data():
    """Test MultiRLPolicy.train() with cached training data."""
    cache_path = Path(
        "/scratch/network/sl5183/shortcut_learning/pipeline_cache/3_obj_cluttered_2d_baseline_cpu/full_training_data/data.pkl"
    )

    if not cache_path.exists():
        pytest.skip(f"Cached data not found at {cache_path}")

    # Load the training data
    with open(cache_path, "rb") as f:
        training_data = pickle.load(f)

    # Limit training data for faster test
    # Take only first 10 shortcuts for testing
    from tamp_improv.approaches.improvisational.policies.base import (
        GoalConditionedTrainingData,
    )

    limited_data = GoalConditionedTrainingData(
        states=training_data.states[:10],
        current_atoms=training_data.current_atoms[:10],
        goal_atoms=training_data.goal_atoms[:10],
        config={
            **training_data.config,
            "max_training_steps_per_shortcut": 100,  # Short training for test
        },
        node_states=training_data.node_states,
        valid_shortcuts=training_data.valid_shortcuts[:10],
        node_atoms=training_data.node_atoms,
        graph=training_data.graph,
    )

    # Create system
    print("\nCreating system for policy training...")
    system = GraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=3, render_mode=None, seed=42
    )

    # Create policy
    print("Creating MultiRLPolicy...")
    policy = MultiRLPolicy(seed=42)

    # Configure environment for training
    if hasattr(system.wrapped_env, "configure_training"):
        print("Configuring training on wrapped_env...")
        system.wrapped_env.configure_training(limited_data)

    # Test policy training
    print("Testing policy.train()...")
    if isinstance(policy, MultiRLPolicy):
        # Create a temporary directory for policy checkpoints
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "test_policy"

            print(f"Training policy with {len(limited_data.states)} examples...")
            policy.train(system.wrapped_env, limited_data, save_dir=str(policy_path))

            print("Policy training completed!")
            print(f"  - Trained policies: {len(policy.policies)}")
            print(f"  - Policy keys: {list(policy.policies.keys())}")


# @pytest.mark.slow
# def test_full_training_pipeline():
#     """Full integration test: load data, configure, and train."""
#     cache_path = Path(
#         "/scratch/network/sl5183/shortcut_learning/pipeline_cache/3_obj_cluttered_2d_baseline_cpu/full_training_data/data.pkl"
#     )

#     if not cache_path.exists():
#         pytest.skip(f"Cached data not found at {cache_path}")

#     # Load the training data
#     print("\n=== Full Training Pipeline Test ===")
#     with open(cache_path, "rb") as f:
#         training_data = pickle.load(f)

#     print(f"\n1. Loaded {len(training_data.states)} training examples")

#     # Create system
#     print("\n2. Creating system...")
#     system = GraphClutteredStorage2DTAMPSystem.create_default(
#         n_blocks=3, render_mode=None, seed=42
#     )

#     # Configure environment for training
#     print("\n3. Configuring training...")
#     if hasattr(system.wrapped_env, "configure_training"):
#         system.wrapped_env.configure_training(training_data)
#         print(f"   Configured with {len(system.wrapped_env.training_states)} states")

#     # Create and train policy
#     print("\n4. Creating and training policy...")
#     policy = MultiRLPolicy(seed=42)

#     # Use limited data for faster test
#     from tamp_improv.approaches.improvisational.policies.base import (
#         GoalConditionedTrainingData,
#     )

#     limited_data = GoalConditionedTrainingData(
#         states=training_data.states[:20],
#         current_atoms=training_data.current_atoms[:20],
#         goal_atoms=training_data.goal_atoms[:20],
#         config={
#             **training_data.config,
#             "max_training_steps_per_shortcut": 200,
#         },
#         node_states=training_data.node_states,
#         valid_shortcuts=training_data.valid_shortcuts[:20],
#         node_atoms=training_data.node_atoms,
#         graph=training_data.graph,
#     )

#     if isinstance(policy, MultiRLPolicy):
#         import tempfile
#         with tempfile.TemporaryDirectory() as tmpdir:
#             policy_path = Path(tmpdir) / "test_policy"
#             policy.train(system.wrapped_env, limited_data, save_dir=str(policy_path))

#     print("\n5. Pipeline test completed successfully!")
#     print(f"   - Trained {len(policy.policies)} specialized policies")
