# PlaceOnTarget Fallback Logic - Debugging Summary

## Original Goal
Implement a fallback mechanism in `PlaceOnTarget` skill to prevent pushing obstruction blocks when the target surface is blocked. Instead of pushing obstructions, place the held block back down and let the planner replan.

## Approach
- **Pre-check**: Before placing, check if any obstruction covers >10% of the target surface
- **Fallback mode**: If blocked, place the block at the current x-coordinate (straight down) instead of on the target surface
- **Expected flow**: Pick up target block → detect surface blocked → place back down → planner replans → pick up obstruction → place in garbage → pick up target block again → place on now-clear surface

---

## Bug Timeline & Solutions

### Bug #1: Block Clipping Through Table During Fallback Placement
**Symptom**: When using fallback placement with `surface_y=0.0, surface_height=0.0`, blocks were clipping through the table.

**Root Cause**: Initial implementation tried to place blocks on the ground (y=0), but the table is actually at y=0.05 with height 0.1 (top at y=0.15).

**Solution**: Use the same `surface_y` and `surface_height` for both normal and fallback modes, since the target surface and table are at the same height.

---

### Bug #2: Horizontal Pushing of Obstruction Blocks During Fallback
**Symptom**: Robot was moving horizontally (left) while holding the purple block in fallback mode, pushing the red obstruction block.

**Root Cause**: Using clamped x-position `target_x = np.clip(robot_x, 0.5, 2.5)` caused Phase 2 to try moving horizontally to stay within bounds.

**Solution**: Changed to exact current position: `target_x = p['robot_x']` to prevent any horizontal movement in fallback mode.

---

### Bug #3: Immediate Block Drop - PickUp Never Lifts
**Symptom**:
```
[PickUp] Phase 6-CloseGripper: ... finger_gap=0.280
[Perceiver] target_block_held=True
[PlaceOnTarget] ENTRY: target_block_held=True, robot_y=0.986
[Perceiver] target_block_held=False, finger_gap=0.300  # Block dropped!
```

**Root Cause**: After PickUp Phase 6 closes the gripper (at `robot_y~1.0`), the planner immediately switches to PlaceOnTarget before PickUp can execute Phase 7 (lift to SAFE_Y). PlaceOnTarget receives the block at low height (`robot_y=0.986`), and the fallback `target_y=0.968` is within tolerance (0.05), so it immediately opens the gripper (Phase 4) and drops the block.

**Why PickUp Phase 7 doesn't execute**: The SLAP framework considers an operator complete as soon as its effects are satisfied. When the `Holding` predicate becomes true (after Phase 6), the planner switches to the next operator (PlaceOnTarget) without waiting for Phase 7.

---

### Attempted Solution #3a: Priority Lift in PickUp
**Approach**: Add a priority check at the beginning of PickUp: if holding the block but not at SAFE_Y, lift first.

**Code**:
```python
# In PickUp._get_action_given_objects()
if p['target_block_held'] and p['robot_y'] < self.SAFE_Y - self.POSITION_TOL:
    return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0])
```

**Result**: Doesn't help because once PickUp completes and switches to PlaceOnTarget, PickUp is never called again.

---

### Attempted Solution #3b: Priority Lift in PlaceOnTarget (Unconditional)
**Approach**: Add a priority check in PlaceOnTarget: if holding the block but not at SAFE_Y, lift first before doing anything else.

**Code**:
```python
# In PlaceOnTarget._get_action_given_objects()
if p['target_block_held'] and p['robot_y'] < self.SAFE_Y - self.POSITION_TOL:
    return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0])
```

**Result**: Successfully lifts from `robot_y=0.986` to SAFE_Y=1.5. ✓

**New Problem**: Stuttering during descent!

---

### Bug #4: Stuttering During Descent
**Symptom**:
```
[PlaceOnTarget] Phase 3: Descending (robot_y=1.487 → 1.437)
[PlaceOnTarget] PRIORITY-Lift-WithBlock: (robot_y=1.437 → 1.487)
[PlaceOnTarget] Phase 3: Descending (robot_y=1.487 → 1.437)
[PlaceOnTarget] PRIORITY-Lift-WithBlock: (robot_y=1.437 → 1.487)
```

**Root Cause**: After lifting to SAFE_Y=1.5, PlaceOnTarget proceeds to Phase 3 (descend toward `target_y=0.968`). It descends 0.05 units to `robot_y=1.45`. On the next call, the priority check sees `robot_y < SAFE_Y`, triggers the lift again, bringing robot back to 1.5. Infinite oscillation.

---

### Attempted Solution #4a: Increase LIFT_THRESHOLD
**Approach**: Only lift if robot is SIGNIFICANTLY below SAFE_Y (e.g., more than 0.2 units below).

**Code**:
```python
LIFT_THRESHOLD = 0.2
if p['target_block_held'] and p['robot_y'] < self.SAFE_Y - LIFT_THRESHOLD:
    # lift
```

**Result**:
- Threshold 0.2: Still stutters at `robot_y=1.3` (descends to 1.286, triggers lift)
- Threshold 0.6: Doesn't lift at handoff (0.986 > 0.9)

**Problem**: Can't find a threshold that works for both scenarios:
1. Lift from PickUp handoff (robot_y ≈ 1.0)
2. Don't interrupt descent (robot_y going from 1.5 → 0.968)

---

### Attempted Solution #4b: Check Alignment
**Approach**: Only lift if not aligned. Once aligned, we're in the descent sequence - don't interrupt.

**Code**:
```python
angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
is_aligned = abs(angle_error) <= self.POSITION_TOL
should_lift = p['target_block_held'] and not_at_safe_y and not is_aligned
```

**Result**: Doesn't work! Debug logs show:
```
[PlaceOnTarget] ENTRY: robot_y=0.986
[PlaceOnTarget] Lift check: is_aligned=True, angle_error=-0.008
```

The robot IS already aligned when PickUp hands off (PickUp aligned it during Phase 1), so `should_lift = False` and the lift doesn't trigger. Block gets dropped immediately.

---

### Attempted Solution #4c: Fixed Threshold (WAY_BELOW_SAFE)
**Approach**: Use a fixed threshold (e.g., 1.2) - lift only if below this value.

**Code**:
```python
WAY_BELOW_SAFE = 1.2
if p['target_block_held'] and p['robot_y'] < WAY_BELOW_SAFE:
    # lift
```

**Result**:
- At handoff (`robot_y=0.986 < 1.2`): Triggers lift ✓
- During descent from 1.5: Robot descends 1.5 → 1.45 → 1.4 → 1.35 → 1.3 → 1.25 → **1.2** → 1.15
- At `robot_y=1.15 < 1.2`: Triggers lift again ✗
- Stuttering resumes!

**Problem**: The descent target is 0.968, which is well below 1.2, so the descent will always cross the threshold and trigger stuttering.

---

## Current Status

**STUCK**: Cannot distinguish between two scenarios without state tracking:
1. **Initial handoff** (robot_y ≈ 1.0): Need to lift to SAFE_Y
2. **Controlled descent** (robot_y going from 1.5 → 0.968): Should NOT interrupt

**Fundamental Issue**: SLAP skills are stateless. We can't track whether "we've already lifted" or "this is an intentional descent."

---

## Potential Next Steps

### Option A: Force PickUp to Complete Phase 7
Somehow prevent the planner from switching away from PickUp until Phase 7 (lift to SAFE_Y) completes. This would require changes to the SLAP framework or operator definitions.

### Option B: Remove Priority Lift, Redesign Fallback
Accept that PlaceOnTarget receives blocks at low height. Redesign the fallback logic to handle this correctly without immediately dropping the block.

### Option C: State Tracking via Predicates
Add a new predicate like `AtSafeHeight(robot)` to the planning domain. Only allow PlaceOnTarget to execute when both `Holding(robot, block)` AND `AtSafeHeight(robot)` are true. This requires changes to:
- Predicates definition
- PickUp operator effects
- PlaceOnTarget operator preconditions
- Perceiver logic

### Option D: Increase CLEARANCE Even More
Instead of 0.15, use a much larger clearance (e.g., 0.5 or more) so that even when receiving the block at `robot_y=0.986`, the `target_y` is significantly different and PlaceOnTarget can execute a meaningful descent before opening the gripper.

---

## Key Learnings

1. **SLAP operator completion is predicate-based**: Operators complete as soon as their effects (predicates) are satisfied, not when the skill returns zeros.

2. **Skills are stateless**: Cannot distinguish between "first time here" vs "continuing from before" without external state.

3. **Tolerance matters**: `POSITION_TOL=0.05` means positions within 0.05 units are considered "the same", which caused PlaceOnTarget to think it was already at the placement height.

4. **Kinematic bodies penetrate**: The KINEMATIC gripper can push through table surfaces during descent, leading to block clipping if descent stops too late.

5. **Alignment is persistent**: Once PickUp aligns the robot, it stays aligned when PlaceOnTarget takes over, making alignment unsuitable as a state indicator.
