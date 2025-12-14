"""Visualization script for PPO distance experiment results.

This script creates plots showing how PPO success rate varies with
the Manhattan distance between initial state and goal cell.

Usage:
    python -m src.tamp_improv.analysis.visualize_distance_results \
        --summary_file results/ppo_distance/summary_20251214_134126.csv \
        --output_file results/ppo_distance/distance_vs_success.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_summary_data(summary_file: str) -> pd.DataFrame:
    """Load summary CSV file into a clean DataFrame.

    The summary file has a multi-level header structure:
    - Row 0: Metric names (success_rate, avg_episode_length, etc.)
    - Row 1: Aggregation types (mean, std, min, max)
    - Row 2: "distance" label
    - Rows 3+: Data with distance as index

    Args:
        summary_file: Path to summary CSV file

    Returns:
        DataFrame with distance as index and hierarchical columns
    """
    # Read with multi-index columns
    df = pd.read_csv(summary_file, header=[0, 1], index_col=0)

    # Skip the "distance" row if it exists (row 2)
    if 'distance' in df.index:
        df = df.drop('distance')

    # Convert index to numeric (distance values)
    df.index = pd.to_numeric(df.index)
    df.index.name = 'distance'

    return df


def plot_success_vs_distance(
    df: pd.DataFrame,
    output_file: str | None = None,
    show: bool = True,
) -> None:
    """Plot success rate vs distance with error bars.

    Args:
        df: DataFrame with summary statistics
        output_file: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Extract success rate data
    success_mean = df[('success_rate', 'mean')]
    success_std = df[('success_rate', 'std')]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with error bars
    ax.errorbar(
        success_mean.index,
        success_mean.values,
        yerr=success_std.values,
        marker='o',
        markersize=8,
        linestyle='-',
        linewidth=2,
        capsize=5,
        capthick=2,
        color='#2E86AB',
        ecolor='#A23B72',
        label='Mean ± Std'
    )

    # Formatting
    ax.set_xlabel('Manhattan Distance (cells)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('PPO Success Rate vs Initial-Goal Distance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)

    # Set x-axis to show integer distances
    ax.set_xticks(success_mean.index)

    # Add legend
    ax.legend(fontsize=10, loc='best')

    # Tight layout
    plt.tight_layout()

    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_comprehensive_metrics(
    df: pd.DataFrame,
    output_file: str | None = None,
    show: bool = True,
) -> None:
    """Plot comprehensive metrics: success rate, episode length, and reward.

    Args:
        df: DataFrame with summary statistics
        output_file: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Success Rate
    ax = axes[0]
    success_mean = df[('success_rate', 'mean')]
    success_std = df[('success_rate', 'std')]
    ax.errorbar(
        success_mean.index,
        success_mean.values,
        yerr=success_std.values,
        marker='o',
        markersize=8,
        linestyle='-',
        linewidth=2,
        capsize=5,
        capthick=2,
        color='#2E86AB',
        ecolor='#A23B72',
    )
    ax.set_xlabel('Distance (cells)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax.set_title('Success Rate vs Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(success_mean.index)

    # Plot 2: Average Episode Length
    ax = axes[1]
    length_mean = df[('avg_episode_length', 'mean')]
    length_std = df[('avg_episode_length', 'std')]
    ax.errorbar(
        length_mean.index,
        length_mean.values,
        yerr=length_std.values,
        marker='s',
        markersize=8,
        linestyle='-',
        linewidth=2,
        capsize=5,
        capthick=2,
        color='#F18F01',
        ecolor='#C73E1D',
    )
    ax.set_xlabel('Distance (cells)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Episode Length', fontsize=11, fontweight='bold')
    ax.set_title('Episode Length vs Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(length_mean.index)

    # Plot 3: Average Reward
    ax = axes[2]
    reward_mean = df[('avg_reward', 'mean')]
    reward_std = df[('avg_reward', 'std')]
    ax.errorbar(
        reward_mean.index,
        reward_mean.values,
        yerr=reward_std.values,
        marker='^',
        markersize=8,
        linestyle='-',
        linewidth=2,
        capsize=5,
        capthick=2,
        color='#06A77D',
        ecolor='#D4A574',
    )
    ax.set_xlabel('Distance (cells)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Reward', fontsize=11, fontweight='bold')
    ax.set_title('Reward vs Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(reward_mean.index)

    # Tight layout
    plt.tight_layout()

    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive plot to {output_file}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_ppo_vs_random_scatter(
    results_df: pd.DataFrame,
    output_file: str | None = None,
    show: bool = True,
) -> None:
    """Plot PPO success vs Random success as a scatter plot.

    Each point represents one cell pair, showing how PPO performs
    compared to random rollouts on the same initial→goal configuration.

    Args:
        results_df: DataFrame with detailed results (not summary)
        output_file: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Check if random metrics are available
    if 'random_success_rate' not in results_df.columns:
        print("Warning: Random policy metrics not found in results. Skipping scatter plot.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get unique distances for color coding
    distances = results_df['distance'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(distances)))
    distance_to_color = dict(zip(distances, colors))

    # Plot each point colored by distance
    for distance in distances:
        subset = results_df[results_df['distance'] == distance]
        ax.scatter(
            subset['random_success_rate'],
            subset['success_rate'],
            c=[distance_to_color[distance]],
            label=f'Distance {distance}',
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=0.5,
        )

    # Plot diagonal line (y=x) for reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='PPO = Random')

    # Formatting
    ax.set_xlabel('Random Policy Success Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('PPO Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('PPO vs Random: Success on Same Cell Pairs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    # Add legend
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    # Add text annotation
    ax.text(
        0.05, 0.95,
        'Above diagonal: PPO outperforms Random\nBelow diagonal: Random outperforms PPO',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Tight layout
    plt.tight_layout()

    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to {output_file}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a formatted summary table.

    Args:
        df: DataFrame with summary statistics
    """
    print("\n" + "="*80)
    print("PPO DISTANCE EXPERIMENT SUMMARY")
    print("="*80 + "\n")

    print(f"{'Distance':<10} {'Success Rate':<20} {'Avg Episode Length':<20} {'Avg Reward':<20}")
    print(f"{'(cells)':<10} {'(mean ± std)':<20} {'(mean ± std)':<20} {'(mean ± std)':<20}")
    print("-" * 80)

    for distance in df.index:
        success_mean = df.loc[distance, ('success_rate', 'mean')]
        success_std = df.loc[distance, ('success_rate', 'std')]
        length_mean = df.loc[distance, ('avg_episode_length', 'mean')]
        length_std = df.loc[distance, ('avg_episode_length', 'std')]
        reward_mean = df.loc[distance, ('avg_reward', 'mean')]
        reward_std = df.loc[distance, ('avg_reward', 'std')]

        print(f"{int(distance):<10} "
              f"{success_mean:.3f} ± {success_std:.3f}      "
              f"{length_mean:.1f} ± {length_std:.1f}      "
              f"{reward_mean:.1f} ± {reward_std:.1f}")

    print("="*80 + "\n")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize PPO distance experiment results"
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        help="Path to summary CSV file (e.g., results/ppo_distance/summary_20251214_134126.csv)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to detailed results CSV file (required for scatter plot)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the main plot (default: same directory as summary with .png extension)",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate comprehensive plot with all metrics",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Generate PPO vs Random scatter plot (requires --results_file)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots (only save to file)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.scatter and not args.results_file:
        parser.error("--scatter requires --results_file")
    if not args.scatter and not args.summary_file:
        parser.error("--summary_file is required for non-scatter plots")

    show = not args.no_show

    # Handle scatter plot (uses detailed results)
    if args.scatter:
        print(f"Loading detailed results from {args.results_file}...")
        results_df = pd.read_csv(args.results_file)

        # Determine output file
        if args.output_file is None:
            results_path = Path(args.results_file)
            output_file = str(results_path.parent / "ppo_vs_random_scatter.png")
        else:
            output_file = args.output_file

        print("\nGenerating PPO vs Random scatter plot...")
        plot_ppo_vs_random_scatter(results_df, output_file=output_file, show=show)

    # Handle summary plots
    else:
        # Load data
        print(f"Loading data from {args.summary_file}...")
        df = load_summary_data(args.summary_file)

        # Print summary table
        print_summary_table(df)

        # Determine output file
        if args.output_file is None:
            summary_path = Path(args.summary_file)
            if args.comprehensive:
                output_file = str(summary_path.parent / "comprehensive_metrics.png")
            else:
                output_file = str(summary_path.parent / "distance_vs_success.png")
        else:
            output_file = args.output_file

        # Generate plots
        if args.comprehensive:
            print("\nGenerating comprehensive metrics plot...")
            plot_comprehensive_metrics(df, output_file=output_file, show=show)
        else:
            print("\nGenerating success rate plot...")
            plot_success_vs_distance(df, output_file=output_file, show=show)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
