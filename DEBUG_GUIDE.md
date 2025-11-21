# Physics Debug Logging Guide

This guide explains how to use the debug logging system to track physics collisions, state transitions, and boundary violations in the DynObstruction2D environment.

## Quick Start

The debug system is **automatically enabled** when you create a `DynObstruction2DTAMPSystem`. No additional setup required!

```python
from src.tamp_improv.benchmarks.dyn_obstruction2d import DynObstruction2DTAMPSystem

# Debug patches are applied automatically
system = DynObstruction2DTAMPSystem.create_default(num_obstructions=2, seed=42)
```

## Debug Flags

Control which types of logging are enabled by editing `src/tamp_improv/benchmarks/debug_physics.py`:

```python
DEBUG_COLLISIONS = True      # Log when objects collide with walls
DEBUG_HELD_OBJECTS = True    # Log DYNAMIC ↔ KINEMATIC transitions
DEBUG_BOUNDS = True          # Log when objects near/outside bounds
DEBUG_SKILL_PHASES = False   # Log skill execution phases (verbose!)
```

## Debug Output Types

### 1. Collision Detection

**`[COLLISION]`** - Triggered when robot parts collide with walls

```
[COLLISION] ROBOT_STATIC: robot hit wall at (0.052, 1.234)
```

**`[ROBOT_REVERT]`** - Triggered when robot reverts to last valid position

```
[ROBOT_REVERT] wall_collision at (3.200, 1.500)
  Held objects also reverted: ['target_block']
```

**What this tells you:**
- Robot tried to penetrate a wall
- Robot position was reverted to last valid state
- Any held objects were also reverted

### 2. Held Object State Transitions

**`[HELD_STATE]`** - Triggered when objects switch between physics modes

```
[HELD_STATE] target_block: DYNAMIC -> KINEMATIC, collision_type=ROBOT
[HELD_STATE] target_block: KINEMATIC -> DYNAMIC, collision_type=DYNAMIC
```

**What this tells you:**
- `DYNAMIC → KINEMATIC`: Object was grasped, now follows gripper
- `KINEMATIC → DYNAMIC`: Object was released, now affected by physics
- Collision type changed to match new state

### 3. Bounds Checking

**`[BOUNDS WARNING]`** - Object within 0.15 units of boundary

```
[BOUNDS WARNING] target_block near right wall: distance=0.082 at (3.154, 0.856)
```

**`[BOUNDS ERROR]`** - Object outside world boundaries!

```
[BOUNDS ERROR] target_block OUT OF BOUNDS at (3.450, 0.856)
  World bounds: x=[0.000, 3.236], y=[0.000, 2.000]
  Violations: x_min=False, x_max=True, y_min=False, y_max=False
```

**What this tells you:**
- Object escaped the physics simulation boundaries
- Which boundaries were violated
- This indicates a bug in collision handling!

### 4. Skill Phases (Optional)

**`[SkillName]`** - Tracks skill execution phases

```
[PickUp] Phase 4: dx=0.050, dy=0.000, ... | robot_x=1.234, block_x=1.567
```

Enable with `DEBUG_SKILL_PHASES = True` for detailed skill debugging.

## Using the Debug System

### Run Test Script

```bash
python test_debug_logging.py
```

This runs random actions and tries to trigger boundary violations.

### In Your Own Code

```python
# The debug system is automatically active!
system = DynObstruction2DTAMPSystem.create_default(num_obstructions=2, seed=42)
obs, info = system.env.reset()

for _ in range(100):
    action = system.env.action_space.sample()
    obs, reward, done, truncated, info = system.env.step(action)
    # Debug messages will print automatically
```

### Interpreting the Logs

**Normal operation (no issues):**
- Occasional `[BOUNDS WARNING]` near boundaries is OK
- `[COLLISION]` + `[ROBOT_REVERT]` means walls are working!
- `[HELD_STATE]` transitions show grasping/releasing

**Problem indicators:**
- `[BOUNDS ERROR]` = Object escaped! This shouldn't happen
- `[ROBOT_REVERT]` with held objects, but block still escapes = timing issue
- Repeated `[BOUNDS WARNING]` in same location = stuck near wall

## Debugging Held Object Escapes

If you see blocks escaping while held, look for this pattern:

```
[HELD_STATE] target_block: DYNAMIC -> KINEMATIC, collision_type=ROBOT
[BOUNDS WARNING] target_block near right wall: distance=0.120 at (3.116, 1.200)
[BOUNDS WARNING] target_block near right wall: distance=0.050 at (3.186, 1.200)
[BOUNDS ERROR] target_block OUT OF BOUNDS at (3.280, 1.200)
```

This shows:
1. Block was grasped (became KINEMATIC)
2. Block approached wall while held
3. Block **escaped** despite collision handlers

**Root cause:** Held KINEMATIC objects are positioned directly, bypassing collision response. The collision handler reverts the *robot*, but the revert might happen too late or the held object position is set after collision detection.

## Advanced: Adding Custom Logging

Add your own debug points by importing the logger:

```python
from tamp_improv.benchmarks.debug_physics import log_bounds_check, log_collision

# In your code:
log_bounds_check("my_object", (x, y), (min_x, max_x, min_y, max_y))
```

## File Reference

- **`src/tamp_improv/benchmarks/debug_physics.py`** - Core logging functions and flags
- **`src/tamp_improv/benchmarks/physics_debug_patch.py`** - Monkey-patches prbench
- **`src/tamp_improv/benchmarks/dyn_obstruction2d.py`** - Auto-applies patches (line 1363-1372)
- **`test_debug_logging.py`** - Test script
- **`DEBUG_GUIDE.md`** - This file

## Troubleshooting

**No debug output appearing:**
1. Check flags in `debug_physics.py` are set to `True`
2. Ensure you're using `DynObstruction2DTAMPSystem.create_default()` (not creating env manually)
3. Look for `[DEBUG_PATCH]` messages during system creation

**Errors during patching:**
```
[WARNING] Failed to apply debug patches: ...
```
The system will still work, but without debug logging. Check the error message for details.

**Too much output:**
Set `DEBUG_SKILL_PHASES = False` to reduce verbosity.
