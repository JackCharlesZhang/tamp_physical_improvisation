# PPO Distance Analysis

This directory contains analysis tools for studying PPO performance on the gridworld environment.

## Files

- **`ppo_distance.py`**: Main experiment script that trains and evaluates PPO policies at different distances
- **`visualize_distance_results.py`**: Visualization script for plotting experiment results

## Experiment Design

### `ppo_distance.py`

The experiment analyzes how PPO's learning performance varies with the Manhattan distance between initial and goal cells.

**Training Strategy:**
- For each distance in the range (e.g., [1, 2, 3, 4, 5, 6]):
  - Sample N cell pairs at that distance (e.g., N=40)
  - **Train a SEPARATE PPO policy for EACH cell pair**
  - Evaluate each policy on the cell pair it was trained on

This means if you test 6 distances with 40 pairs each, you'll train **240 separate PPO policies** (not 6).

**Key Parameters:**
- `num_cells`: Grid size (e.g., 4×4 cells)
- `distance_range`: Which distances to test (e.g., [1, 2, 3, 4, 5, 6])
- `num_training_pairs_per_distance`: How many cell pairs to sample per distance (each gets its own policy)
- `total_timesteps`: Training timesteps per policy (e.g., 200,000)

**Output:**
- Detailed CSV with one row per trained policy
- Summary CSV with statistics aggregated by distance
- Model checkpoints for each trained policy

### Usage

```bash
# Basic usage with defaults
python -m src.tamp_improv.analysis.ppo_distance

# Custom configuration
python -m src.tamp_improv.analysis.ppo_distance \
    --num_cells 5 \
    --distances 1 2 3 4 \
    --num_training_pairs 10 \
    --total_timesteps 100000 \
    --seed 42
```

### `visualize_distance_results.py`

Creates visualizations from experiment results, including PPO vs Random comparisons.

**Usage:**

```bash
# Simple success rate plot
python -m src.tamp_improv.analysis.visualize_distance_results \
    --summary_file results/ppo_distance/summary_TIMESTAMP.csv

# Comprehensive plot with all metrics
python -m src.tamp_improv.analysis.visualize_distance_results \
    --summary_file results/ppo_distance/summary_TIMESTAMP.csv \
    --comprehensive

# PPO vs Random scatter plot (requires detailed results file)
python -m src.tamp_improv.analysis.visualize_distance_results \
    --results_file results/ppo_distance/results_TIMESTAMP.csv \
    --scatter

# Save without displaying
python -m src.tamp_improv.analysis.visualize_distance_results \
    --summary_file results/ppo_distance/summary_TIMESTAMP.csv \
    --no_show
```

**Scatter Plot:**
The scatter plot shows PPO success rate (y-axis) vs Random success rate (x-axis) for each cell pair:
- Points above the diagonal: PPO outperforms random
- Points below the diagonal: Random outperforms PPO
- Points are colored by distance for easy identification

## Interpretation

The experiment answers: **How does the distance between initial and goal cells affect PPO's ability to learn the task?**

Expected patterns:
- **Shorter distances** may be easier to learn (fewer steps needed)
- **Longer distances** may be harder (more exploration needed, longer episodes)
- Results show whether PPO's success rate increases, decreases, or remains constant with distance
