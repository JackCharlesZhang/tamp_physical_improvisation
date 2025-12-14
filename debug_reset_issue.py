"""Debug script to identify observation_objects mutation issue."""

import pickle
from pathlib import Path

from tamp_improv.benchmarks.clutteredstorage_system import (
    GraphClutteredStorage2DTAMPSystem,
)

def main():
    cache_path = Path(
        "/scratch/network/sl5183/shortcut_learning/pipeline_cache/3_obj_cluttered_2d_baseline_cpu/full_training_data/data.pkl"
    )

    if not cache_path.exists():
        print(f"ERROR: Cached data not found at {cache_path}")
        return

    # Load the training data
    print("\n" + "="*80)
    print("STEP 1: Loading training data")
    print("="*80)
    with open(cache_path, "rb") as f:
        training_data = pickle.load(f)

    print(f"Loaded {len(training_data.states)} training states")
    if len(training_data.states) > 0:
        print(f"First state shape: {training_data.states[0].shape if hasattr(training_data.states[0], 'shape') else type(training_data.states[0])}")

    # Create system
    print("\n" + "="*80)
    print("STEP 2: Creating system")
    print("="*80)
    system = GraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=3, render_mode=None, seed=42
    )

    print("\n" + "="*80)
    print("STEP 3: After creating system")
    print("="*80)
    print(f"wrapped_env type: {type(system.wrapped_env)}")
    print(f"base env type: {type(system.env)}")
    if hasattr(system.env, 'observation_objects'):
        print(f"system.env.observation_objects: {system.env.observation_objects}")
        print(f"system.env.observation_objects length: {len(system.env.observation_objects)}")

    # Configure training
    print("\n" + "="*80)
    print("STEP 4: Configuring training")
    print("="*80)
    system.wrapped_env.configure_training(training_data)

    print("\n" + "="*80)
    print("STEP 5: After configuring training")
    print("="*80)
    if hasattr(system.env, 'observation_objects'):
        print(f"system.env.observation_objects: {system.env.observation_objects}")
        print(f"system.env.observation_objects length: {len(system.env.observation_objects)}")

    # Test reset
    print("\n" + "="*80)
    print("STEP 6: Testing first reset")
    print("="*80)
    obs, info = system.wrapped_env.reset()
    print(f"Reset successful! Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

    # Test second reset
    print("\n" + "="*80)
    print("STEP 7: Testing second reset")
    print("="*80)
    obs, info = system.wrapped_env.reset()
    print(f"Reset successful! Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

if __name__ == "__main__":
    main()
