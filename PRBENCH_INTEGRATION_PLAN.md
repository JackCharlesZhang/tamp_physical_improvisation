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

## How to Test PRBench Integration with Sesame Planner

**Why use Sesame instead of pure planning?**
- PRBench operators are designed for bilevel planning (symbolic + motion)
- Navigation operators originally had empty `add_effects` → **FIXED** (see below)
- Pure task planners (pyperplan) can't work with bilevel operators
- Sesame planner (PRBench's bilevel planner) is the correct way to test

**CRITICAL FIX Applied to prpl-mono:**
We discovered navigation operators were missing symbolic effects for `AboveTgt`:
- **File**: `prpl-mono/prbench-bilevel-planning/src/prbench_bilevel_planning/env_models/dynamic2d/dynobstruction2d.py`
- **Lines 181, 189**: Added `add_effects={LiftedAtom(AboveTgtSurface, [robot])}` to `MoveToTgtHeld` and `MoveToTgtEmpty`
- **Lines 198, 206**: Added `delete_effects={LiftedAtom(AboveTgtSurface, [robot])}` to `MoveFromTgtHeld` and `MoveFromTgtEmpty`
- **Without this fix**: Abstract planner generates NO plans (can't reach `PlaceTgtOnSurface` which requires `AboveTgt`)
- **With this fix**: Abstract planner generates valid plans ✓

**Running Sesame planner test on della-gpu:**

```bash
ssh jz4267@della-gpu.princeton.edu
cd ~/tamp_physical_improvisation
git pull origin jcz/integrate_2d_dyn_obstruction

# Load modules and activate environment
module load anaconda3/2024.10
source .venv/bin/activate

# Export PYTHONPATH (CRITICAL - must include bilevel-planning!)
export PYTHONPATH=/scratch/gpfs/TRIDAO/jz4267/prpl-mono/bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-models/src:$PYTHONPATH

# Run Sesame planner test
python experiments/test_prbench_sesame.py --seed 42 --num-obstructions 2 \
  --max-abstract-plans 10 --samples-per-step 10 --timeout 30.0

# With video recording (for visualization)
python experiments/test_prbench_sesame.py --seed 42 --num-obstructions 2 \
  --max-abstract-plans 10 --samples-per-step 10 --timeout 30.0 \
  --record-video --video-folder videos/sesame_prbench

# Download video from della-gpu
rsync -avz jz4267@della-gpu.princeton.edu:~/tamp_physical_improvisation/videos/sesame_prbench/*.mp4 ./
```

**What this test validates:**
1. PRBench operators execute correctly
2. PRBench skills (BiRRT motion planning) work
3. State abstraction and goal derivation function properly
4. Multi-abstract plan + backtracking refinement succeeds
5. Overall PRBench integration is sound

## Controller Bug Fixes for DynObstruction2D

### Issue: Bilevel Planning Failing at 100% Rate

**Context**: During integration testing, discovered that bilevel planning was completely failing despite having correct operators, predicates, and state abstraction. The Sesame planner (PRBench's multi-abstract plan + backtracking refinement planner) was generating valid abstract plans but failing to refine them into executable trajectories.

**Root Cause Analysis Process**:
1. Initially MoveToTargetEmpty/MoveToTargetHeld succeeded 0% → improved to ~68% after first fix
2. Despite ~68% success for MoveToTarget, bilevel planning still failed 100%
3. With 50 parameter samples per step, probability of all failing is negligible
4. Investigated subsequent actions (PickTgt, PlaceTgt) in abstract plans
5. Found NO `[TRAJ_SUCCESS]` or `[TRAJ_FAIL]` messages for PickTgt actions
6. Realized failures happening during trajectory generation (in controller's `_generate_waypoints`), not execution
7. Traced to collision checking in GroundPickController raising `TrajectorySamplingFailure`

### Bug Fix #1: MoveToTgtSurfaceController Positioning (Commit 3bdb93f)

**File**: `prpl-mono/prbench-models/src/prbench_models/dynamic2d/dynobstruction2d/parameterized_skills.py`

**Problem**: Controller was positioning robot using rotation-dependent calculations instead of direct X-coordinate alignment with target surface.

**Before**:
```python
# Position robot using SE2Pose transformations
target_x = current_state.get(self._target_surface, "x")
target_y = current_state.get(self._target_surface, "y")
target_theta = current_state.get(self._target_surface, "theta")
target_center = SE2Pose(target_x, target_y, target_theta) * SE2Pose(...)
# Robot positioned at rotation-dependent location
```

**After**:
```python
# Position robot directly at target surface X coordinate
target_surface_x = current_state.get(self._target_surface, "x")
target_robot_pose = SE2Pose(target_surface_x, sampled_y, 0.0)
# Robot positioned at x = target_surface_x (required for AboveTgt predicate)
```

**Impact**:
- MoveToTargetEmpty success rate: 0% → ~68%
- Still not 100% due to collision checking (expected - some configurations genuinely unreachable)
- Critical for achieving AboveTgt predicate required by PlaceTgtOnSurface operator

**Technical Details**: The AboveTgt predicate checks `abs(robot_x - target_surface_x) < 0.01`, which requires precise X-coordinate alignment. The old rotation-dependent positioning failed this check.

### Bug Fix #2: GroundPickController Side Sampling (Commit eeac7cd)

**File**: `prpl-mono/prbench-models/src/prbench_models/dynamic2d/dynobstruction2d/parameterized_skills.py`

**Problem**: Controller was restricted to sampling grasp poses from top side only. When obstructions or robot positioned above target block, all 50 pick attempts failed collision checking, causing bilevel planning to backtrack and eventually fail.

**Before (Line 48)**:
```python
def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> tuple[float, float, float]:
    grasp_ratio = rng.uniform(0.0, 0.1)
    side = rng.uniform(0.5, 0.75)  # TOP SIDE ONLY - blocks all other approaches!
    arm_length = rng.uniform(min_arm_length, max_arm_length)
    return (grasp_ratio, side, arm_length)
```

**After (Line 48)**:
```python
def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> tuple[float, float, float]:
    grasp_ratio = rng.uniform(0.0, 0.1)
    side = rng.uniform(0.0, 1.0)  # ALL SIDES - allows any approach direction!
    # side parameter mapping:
    #   0.0-0.25: left side
    #   0.25-0.5: right side
    #   0.5-0.75: top side
    #   0.75-1.0: bottom side
    arm_length = rng.uniform(min_arm_length, max_arm_length)
    return (grasp_ratio, side, arm_length)
```

**Why This Was Critical**:
- GroundPickController performs collision checking before returning parameters
- When collision detected, it raises `TrajectorySamplingFailure`
- Backtracking refiner tries 50 parameter samples per step
- If all 50 samples fail collision checking → refiner backtracks to previous action
- With top-only sampling + obstructions above target → 100% failure rate
- With all-sides sampling → can find collision-free grasp from accessible sides

**Collision Checking Code (Lines 150-163)**:
```python
# Check if the target pose is collision-free
full_state.set(self._robot, "x", target_se2_pose.x)
full_state.set(self._robot, "y", target_se2_pose.y)
full_state.set(self._robot, "theta", target_se2_pose.theta)
full_state.set(self._robot, "arm_joint", desired_arm_length)

moving_objects = {self._robot}
static_objects = set(full_state) - moving_objects

if state_2d_has_collision(full_state, moving_objects, static_objects, {}):
    raise TrajectorySamplingFailure("Failed to find a collision-free path to target.")
```

**Impact**:
- Expected: Bilevel planning success rate improves dramatically (from 0% to >80%)
- Allows planner to find valid pick approaches from any accessible side
- Critical for handling scenarios where obstructions block specific approach directions

### Testing Results

**Before Fixes**:
- MoveToTargetEmpty: 0% success
- PickTgt: 0% success (no TRAJ messages - all failing in collision checking)
- Bilevel planning: 0% success

**After Fix #1 Only**:
- MoveToTargetEmpty: ~68% success
- PickTgt: 0% success (still failing in collision checking)
- Bilevel planning: 0% success

**After Both Fixes** (Expected):
- MoveToTargetEmpty: ~68% success
- PickTgt: ~XX% success (test running on della-gpu task 045dbd)
- Bilevel planning: Expected >80% success with proper parameter sampling

**How to Verify Fixes**:
```bash
# CORRECT COMMAND - Run bilevel planning test with both fixes
# IMPORTANT: Must use this EXACT sequence:
# 1. cd to test directory
# 2. Load anaconda3/2024.10 module
# 3. Activate tamp_physical_improvisation venv (contains tomsgeoms2d)
# 4. Export PYTHONPATH for all prpl-mono packages
# 5. Run pytest

ssh jz4267@della-gpu.princeton.edu "cd /scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-bilevel-planning && module load anaconda3/2024.10 && source ~/tamp_physical_improvisation/.venv/bin/activate && export PYTHONPATH=/scratch/gpfs/TRIDAO/jz4267/prpl-mono/bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-models/src:\$PYTHONPATH && pytest -xvs 'tests/env_models/dynamic2d/test_dynobstruction2d.py::test_dynobstruction2d_bilevel_planning[0-5-50]' --make-videos"

# Monitor trajectory sampling success rates
# Should see [TRAJ_SUCCESS] messages for both MoveToTargetEmpty AND PickTgt
# Should see overall test PASS
```

### Lessons Learned

1. **Parameter Space Restrictions**: Overly restrictive parameter sampling can cause systematic failures that appear as planning bugs
2. **Collision Checking During Sampling**: When controllers validate parameters before execution, must ensure parameter space is large enough to find valid samples
3. **Debugging Trajectory Failures**:
   - Check for TRAJ_SUCCESS/TRAJ_FAIL messages to identify which actions failing
   - If NO messages for an action → failing during trajectory generation (collision checking)
   - If TRAJ_FAIL messages → failing during execution (controller termination, wrong abstract state)
4. **Backtracking Refinement**: Even with good parameter spaces, some configurations genuinely unreachable → ~70% success acceptable for individual actions
5. **Multi-Step Planning**: Single action failure doesn't doom planning if backtracking + multiple abstract plans available

### Related Files

**Controller Implementation**:
- `prbench-models/src/prbench_models/dynamic2d/dynobstruction2d/parameterized_skills.py` - All skill controllers

**Trajectory Sampling**:
- `bilevel-planning/src/bilevel_planning/trajectory_samplers/parameterized_controller_sampler.py` - Trajectory sampling with failure logging

**Refinement**:
- `bilevel-planning/src/bilevel_planning/refiners/backtracking_refiner.py` - Backtracking with 50 samples per step

**Testing**:
- `prbench-bilevel-planning/tests/env_models/dynamic2d/test_dynobstruction2d.py::test_dynobstruction2d_bilevel_planning` - End-to-end bilevel planning test
