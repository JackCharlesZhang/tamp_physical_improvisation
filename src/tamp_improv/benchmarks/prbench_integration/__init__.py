"""PRBench integration for SLAP.

This package provides adapters to use PRBench components (environments,
perception, motion planning) with SLAP's shortcut learning pipeline.
"""

from tamp_improv.benchmarks.prbench_integration.perceiver import PRBenchPerceiver
from tamp_improv.benchmarks.prbench_integration.utils import (
    PRBenchPredicateContainer,
)

__all__ = [
    "PRBenchPerceiver",
    "PRBenchPredicateContainer",
]
