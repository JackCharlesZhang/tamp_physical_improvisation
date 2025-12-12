# PRBench + SLAP Integration Plan

## Goal
Integrate PRBench's clean perception and motion planning with SLAP's shortcut learning pipeline for the Dynamic Obstruction 2D environment.

## Why This Integration?

### Problems with SLAP's Original Implementation
- **Bad counting predicates**: `TwoObstructionsBlocking`, `OneObstructionBlocking`, `Clear` - overly complex
- **Manual observation parsing**: Hard-coded indices like `obs[0]`, `obs[21] > 0.5`
- **Custom geometry checks**: Reimplemented collision/containment logic with magic thresholds
- **7 predicates total** when PRBench uses only 4

### PRBench's Advantages (Ground Truth)
- **4 simple predicates**: `HandEmpty`, `HoldingTgt`, `HoldingObstruction`, `OnTgtSurface`
- **Clean state abstraction**: Uses `state_abstractor` function - no manual parsing
- **PRBench's geometric utilities**: `get_suctioned_objects()`, `is_on()` - battle-tested
- **BiRRT motion planning**: Bidirectional RRT for collision-free paths
- **Generic operators**: Work for any number of obstructions (no counting!)

## Integration Strategy

### What We Keep from PRBench
✅ `state_abstractor` - perception (ObjectCentricState → RelationalAbstractState)
✅ `goal_deriver` - goal extraction
✅ 4 simple predicates - no counting logic
✅ Generic operators - PickTgt, PickObstruction, PlaceTgt, PlaceObstruction
✅ BiRRT motion planning - collision-free paths
✅ PRBench's geometric utilities

### What We Keep from SLAP
✅ Shortcut learning pipeline - the core algorithm
✅ Graph-based RL training - learning which edges to shortcut
✅ `ImprovisationalTAMPApproach` - main training loop
✅ `ShortcutSignature` - transfer learning mechanism
✅ Planning graph construction - nodes + edges

### What We Build (Adapters)
🔧 **Phase 1: Core Adapters** ✓ COMPLETED
- `PRBenchPredicateContainer` - wraps PRBench's predicate set for SLAP's access pattern
- `PRBenchPerceiver` - wraps state_abstractor/goal_deriver as SLAP's Perceiver interface

🔧 **Phase 2: Skill Adapter** (NEXT)
- `PRBenchSkill` - wraps PRBench's ParameterizedController as SLAP's LiftedOperatorSkill
  - **Challenge**: PRBench controllers use `step()` (no obs argument), SLAP expects `get_action(obs)`
  - **Solution**: Sample parameters once in `__init__`, store them, return actions via `get_action(obs)`

🔧 **Phase 3: System Integration**
- `PRBenchSLAPSystem` - main system that combines:
  - PRBench's operators + motion planning
  - SLAP's shortcut learning
  - Proper environment wrapping (VectorizedDynObstruction2DEnv)

🔧 **Phase 4: Training Script**
- New training script that uses `PRBenchSLAPSystem`
- Inherits all SLAP training logic (graph construction, shortcut identification, RL training)

## Technical Details

### Critical Incompatibility Point
**File**: `tamp_physical_improvisation/src/tamp_improv/approaches/improvisational/base.py:728`
```python
def _execute_edge(self, ...):
    ...
    act = skill.get_action(curr_aug_obs)  # SLAP expects obs → action
```

**Issue**: PRBench's `ParameterizedController.step()` takes no observation argument
**Solution**: PRBenchSkill will wrap controllers and provide the `get_action(obs)` interface

### Environment Wrapper Choice
- **Use**: `DynObstruction2DEnv` (returns `ObjectCentricBoxSpace`)
  - Inherits from `ConstantObjectPRBenchEnv`
  - Wraps `ObjectCentricDynObstruction2DEnv` with fixed object ordering
- **Not**: `ObjectCentricDynObstruction2DEnv` (returns `ObjectCentricStateSpace`)
- **Reason**: `create_bilevel_planning_models()` requires `ObjectCentricBoxSpace`

### Predicate Mapping
```
SLAP (7 predicates)                    PRBench (4 predicates)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HandEmpty                        →     HandEmpty
HoldingTargetBlock               →     HoldingTgt
HoldingObstruction               →     HoldingObstruction
TargetBlockOnTargetSurface       →     OnTgtSurface
TwoObstructionsBlocking          →     [eliminated - no counting!]
OneObstructionBlocking           →     [eliminated - no counting!]
Clear                            →     [eliminated - no counting!]
```

### Operator Mapping
```
SLAP (6 operators - state-specific)    PRBench (4 operators - generic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PickTargetBlock                  →     PickTgt
PlaceTargetBlockOnSurface        →     PlaceTgt
PickObstructionWhenOneBlocking   →     PickObstruction (generic!)
PickObstructionWhenTwoBlocking   →     [eliminated - PickObstruction handles all]
PlaceObstructionWhenOneRemaining →     PlaceObstruction (generic!)
PlaceObstructionWhenNoneRemaining→     [eliminated - PlaceObstruction handles all]
```

## Implementation Phases

### ✅ Phase 1: Core Adapters (COMPLETED)
**Files Created**:
- `src/tamp_improv/benchmarks/prbench_integration/__init__.py`
- `src/tamp_improv/benchmarks/prbench_integration/perceiver.py`
- `src/tamp_improv/benchmarks/prbench_integration/utils.py`
- `tests/test_prbench_integration.py`

**Tests**:
- ✓ PRBenchPredicateContainer initialization
- ✓ Predicate access by name (`container["HandEmpty"]`)
- ✓ Conversion to set (`container.as_set()`)
- ✓ Membership checking (`"HandEmpty" in container`)
- ✓ PRBenchPerceiver initialization
- ✓ Perceiver reset (returns objects, atoms, goal)
- ✓ Perceiver step (returns current atoms)
- ✓ Verification of no counting predicates

**Status**: Ready to run tests on della-gpu after git pull

### 🔧 Phase 2: Skill Adapter (NEXT)
**To Create**:
- `src/tamp_improv/benchmarks/prbench_integration/skills.py`
  - `PRBenchSkill(LiftedOperatorSkill)` class
  - Wraps `ParameterizedController` with `get_action(obs)` interface
  - Handles parameter sampling and action generation

**Key Design**:
```python
class PRBenchSkill(LiftedOperatorSkill):
    def __init__(self, controller: ParameterizedController, operator: Operator):
        self.controller = controller
        self.operator = operator
        self._params = None
        self._done = False

    def reset(self, obs, objects):
        # Sample parameters for this execution
        self._params = self.controller.sample_parameters(obs, objects)
        self.controller.reset(obs, self._params)
        self._done = False

    def get_action(self, obs):
        # PRBench controllers use step() without obs
        # We call it and return the action
        if self._done:
            return None  # or raise exception
        action = self.controller.step()
        return action

    def is_complete(self, obs):
        # Check if controller is done
        return self.controller.is_done()
```

### 🔧 Phase 3: System Integration
**To Create**:
- `src/tamp_improv/benchmarks/prbench_integration/system.py`
  - `PRBenchSLAPSystem(BaseTAMPSystem)` class
  - Combines PRBench components with SLAP's interface

**Key Design**:
```python
class PRBenchSLAPSystem(BaseTAMPSystem):
    def __init__(self, num_obstructions: int = 2):
        # Create PRBench environment
        self.env = DynObstruction2DEnv(num_obstructions)

        # Create PRBench models
        self.sesame_models = create_bilevel_planning_models(
            self.env.observation_space,
            self.env.action_space,
            num_obstructions
        )

        # Create adapters
        self.predicates = PRBenchPredicateContainer(self.sesame_models.predicates)
        self.perceiver = PRBenchPerceiver(
            self.sesame_models.state_abstractor,
            self.sesame_models.goal_deriver
        )

        # Create skills from PRBench controllers
        self.skills = {
            op.name: PRBenchSkill(controller, op)
            for op, controller in zip(
                self.sesame_models.operators,
                self.sesame_models.controllers
            )
        }
```

### 🔧 Phase 4: Training Script
**To Create**:
- `experiments/train_prbench_dyn_obstruction2d.py`
  - Uses `PRBenchSLAPSystem`
  - Configures SLAP's training pipeline
  - Same training loop as original SLAP

**Key Configuration**:
```python
# Create system
system = PRBenchSLAPSystem(num_obstructions=2)

# Create approach (SLAP's training pipeline)
approach = ImprovisationalTAMPApproach(
    system=system,
    shortcut_discovery="rollout",  # or "topology"
    ...
)

# Train
approach.train(num_episodes=1000)
```

## No Modifications to prpl-mono!
All adapters live in `tamp_physical_improvisation/src/tamp_improv/benchmarks/prbench_integration/`
- We import from PRBench as-is
- We wrap their components to match SLAP's interface
- Clean separation of concerns

## Key Files Reference

### PRBench Files (Read-only)
- `prbench/src/prbench/envs/dynamic2d/dyn_obstruction2d.py` - Environment
- `prbench-bilevel-planning/src/prbench_bilevel_planning/env_models/dynamic2d/dynobstruction2d.py` - Models
- `prbench-models/src/prbench_models/dynamic2d/utils.py` - BiRRT motion planning

### SLAP Files (Modified/Created)
- `src/tamp_improv/benchmarks/prbench_integration/` - Our adapter package
- `tests/test_prbench_integration.py` - Integration tests
- `experiments/train_prbench_dyn_obstruction2d.py` - Training script (to create)

### SLAP Core (Unchanged)
- `src/tamp_improv/approaches/improvisational/base.py` - Main training loop
- `src/tamp_improv/approaches/improvisational/graph_training.py` - Shortcut discovery
- All SLAP training infrastructure remains the same

## Testing Strategy

### Unit Tests (Phase 1) ✓
- Test each adapter in isolation
- Verify correct interface implementation
- Check predicate/operator mappings

### Integration Tests (Phase 2-3)
- Test full system initialization
- Test skill execution
- Test planning graph construction with PRBench components

### End-to-End Tests (Phase 4)
- Run full training episode
- Verify shortcuts are learned
- Compare performance with original SLAP implementation

## Success Criteria
1. ✓ Phase 1 tests pass
2. PRBench's 4 predicates used throughout (no counting predicates)
3. Skills execute successfully using PRBench controllers
4. Planning graph constructs correctly with PRBench operators
5. SLAP's shortcut learning trains successfully
6. Learned shortcuts improve task completion

## Current Status
- **Phase 1**: ✅ COMPLETED - Core adapters (PRBenchPerceiver, PRBenchPredicateContainer)
- **Phase 2**: ✅ COMPLETED - Skill adapter (PRBenchSkill)
- **Phase 3**: ✅ COMPLETED - System integration (BasePRBenchSLAPSystem, PRBenchSLAPSystem)
- **Phase 4**: ✅ COMPLETED - Training script integration
- **Status**: Ready for training!

## How to Run Tests on della-gpu

**IMPORTANT**: Must set complete PYTHONPATH including `bilevel-planning/src`!

```bash
ssh jz4267@della-gpu.princeton.edu
cd ~/tamp_physical_improvisation
git pull origin jcz/integrate_2d_dyn_obstruction

# Load modules and activate environment
module load anaconda3/2024.10
source .venv/bin/activate

# Export PYTHONPATH (CRITICAL - must include bilevel-planning!)
export PYTHONPATH=/scratch/gpfs/TRIDAO/jz4267/prpl-mono/bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-models/src:$PYTHONPATH

# Run all integration tests
python -m pytest tests/test_prbench_integration.py -v -s

# Run specific phase tests
python -m pytest tests/test_prbench_integration.py::TestPRBenchPredicateContainer -v -s
python -m pytest tests/test_prbench_integration.py::TestPRBenchPerceiver -v -s
python -m pytest tests/test_prbench_integration.py::TestPRBenchSkill -v -s
python -m pytest tests/test_prbench_integration.py::TestPRBenchSLAPSystem -v -s
```

**Note**: The `/scratch/gpfs/TRIDAO/jz4267/prpl-mono` folder on della-gpu is identical to local `~/Desktop/slapo/prpl-mono`

## How to Run Training on della-gpu

**Training with PRBench components + SLAP shortcut learning:**

```bash
ssh jz4267@della-gpu.princeton.edu
cd ~/tamp_physical_improvisation
git pull origin jcz/integrate_2d_dyn_obstruction

# Load modules and activate environment
module load anaconda3/2024.10
source .venv/bin/activate

# Export PYTHONPATH (CRITICAL - must include bilevel-planning!)
export PYTHONPATH=/scratch/gpfs/TRIDAO/jz4267/prpl-mono/bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-models/src:$PYTHONPATH

# Run training with PRBench + SLAP
python experiments/slap_train.py --config-name prbench_dyn_obstruction2d

# Compare with original SLAP implementation (for ablation)
python experiments/slap_train.py --config-name dyn_obstruction2d
```

**What the training does:**
1. Creates `PRBenchSLAPSystem` with PRBench's 5 predicates, 9 operators, 9 skills
2. Uses SLAP's existing `train_and_evaluate()` function (NO modifications!)
3. Discovers shortcuts via rollouts (topology-based or random)
4. Trains RL policies for each shortcut using MultiRL
5. Saves trained policies to `trained_policies/prbench_multi_rl/`

**Configuration files:**
- `experiments/configs/prbench_dyn_obstruction2d.yaml` - PRBench + SLAP config
- `experiments/configs/dyn_obstruction2d.yaml` - Original SLAP config (for comparison)
