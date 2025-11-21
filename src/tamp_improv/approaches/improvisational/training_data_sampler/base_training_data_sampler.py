from typing import Any, TypeVar, Optional, Generic
import numpy as np
import abc
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


ObsType = TypeVar("ObsType")

class BaseTrainingDataSampler(Generic[ObsType], abc.ABC):

    def __init__(self, 
        all_state_pairs: list[tuple[ObsType, ObsType]],
        system: ImprovisationalTAMPSystem,
        planning_graph: Optional[PlanningGraph],
        config: dict[str, Any],
        rng: np.random.Generator,
    ):
        self.all_state_pairs: list[tuple[ObsType, ObsType]] = all_state_pairs
        self.system = system
        self.planning_graph = planning_graph
        self.config = config
        self.rng = rng

    @abc.abstractmethod
    def sample(self) -> list[tuple[ObsType, ObsType]]:
        """Sample training pairs for distance heuristic training."""
        pass
