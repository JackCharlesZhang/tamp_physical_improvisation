"""Debug script to understand observation structure."""

from tamp_improv.benchmarks.dyn_obstruction2d import BaseDynObstruction2DTAMPSystem
from prbench.envs.dynamic2d.object_types import Dynamic2DRobotEnvTypeFeatures

# Create the system
tamp_system = BaseDynObstruction2DTAMPSystem.create_default(
    num_obstructions=2, render_mode="rgb_array", seed=42
)

env = tamp_system.env
obs, info = env.reset()

print("=" * 80)
print("OBSERVATION STRUCTURE DEBUG")
print("=" * 80)

print(f"\nObservation shape: {obs.shape}")
print(f"Observation space: {env.observation_space}")

print(f"\n{'Object':<25} {'Type':<20} {'# Features':<15}")
print("-" * 60)

total_expected = 0
for obj in env._constant_objects:
    features = Dynamic2DRobotEnvTypeFeatures.get(obj.type, [])
    print(f"{obj.name:<25} {obj.type.name:<20} {len(features):<15}")
    total_expected += len(features)

print("-" * 60)
print(f"{'TOTAL EXPECTED':<25} {'':<20} {total_expected:<15}")
print(f"{'ACTUAL OBSERVATION':<25} {'':<20} {obs.shape[0]:<15}")
print(f"{'DIFFERENCE':<25} {'':<20} {obs.shape[0] - total_expected:<15}")

# Show feature details for each object
print("\n" + "=" * 80)
print("DETAILED FEATURE BREAKDOWN")
print("=" * 80)
for obj in env._constant_objects:
    features = Dynamic2DRobotEnvTypeFeatures.get(obj.type, [])
    print(f"\n{obj.name} ({obj.type.name}):")
    print(f"  Features ({len(features)}): {features}")

print("\n" + "=" * 80)
