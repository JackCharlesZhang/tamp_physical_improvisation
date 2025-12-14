# GraphClutteredStorage2DTAMPSystem Initialization Flow

## Problem Overview

The `GraphClutteredStorage2DTAMPSystem` class has **diamond inheritance**, which was causing the base environment to be created twice with different configurations:

```
        BaseTAMPSystem
              |
     +--------+--------+
     |                 |
ImprovisationalTAMPSystem  BaseGraphClutteredStorage2DTAMPSystem
     |                 |
     +--------+--------+
              |
GraphClutteredStorage2DTAMPSystem
```

### Original Issue
When calling both parent constructors explicitly:
- First call to `ImprovisationalTAMPSystem.__init__()` created env with default n_blocks=1
- Second call to `BaseGraphClutteredStorage2DTAMPSystem.__init__()` created env with n_blocks=3
- Result: `wrapped_env` wrapped the wrong environment (1-block), while `self.env` pointed to 3-block env

### Solution
Use **cooperative multiple inheritance** with `super()` to ensure each `__init__` is called exactly once.

## Method Resolution Order (MRO)

Python's MRO for `GraphClutteredStorage2DTAMPSystem`:
```
0: GraphClutteredStorage2DTAMPSystem
1: ImprovisationalTAMPSystem
2: BaseGraphClutteredStorage2DTAMPSystem
3: BaseTAMPSystem
4: Generic
5: ABC
6: object
```

## Initialization Order (Step-by-Step)

When calling:
```python
system = GraphClutteredStorage2DTAMPSystem.create_default(n_blocks=3, seed=42)
```

### 1. GraphClutteredStorage2DTAMPSystem.__init__()
**File:** [clutteredstorage_system.py:449-471](src/tamp_improv/benchmarks/clutteredstorage_system.py#L449-L471)

```python
def __init__(self, planning_components, n_blocks=1, seed=None, render_mode=None):
    # Set attributes BEFORE calling super().__init__()
    # so they're available when _create_env() is called
    self.n_blocks = n_blocks  # Sets n_blocks = 3
    self._render_mode = render_mode

    # Use cooperative multiple inheritance
    super().__init__(planning_components, seed=seed, render_mode=render_mode)
```

**Actions:**
- Sets `self.n_blocks = 3`
- Sets `self._render_mode`
- Calls `super().__init__()` → Goes to next in MRO

---

### 2. ImprovisationalTAMPSystem.__init__()
**File:** [base.py:111-122](src/tamp_improv/benchmarks/base.py#L111-L122)

```python
def __init__(self, planning_components, seed=None, render_mode=None):
    self._render_mode = render_mode
    super().__init__(planning_components, seed=seed, render_mode=render_mode)  # Line 119
    self.wrapped_env = self._create_wrapped_env(planning_components)          # Line 120
    if seed is not None:
        self.wrapped_env.reset(seed=seed)
```

**Actions:**
- Line 119: Calls `super().__init__()` → Goes to next in MRO (completes entire chain)
- **Line 120: Creates `self.wrapped_env = self._create_wrapped_env()`** ← **ImprovWrapper created here!**
- Line 121-122: Resets wrapped_env with seed

**Important:** By the time line 120 executes, the entire super() chain has completed, so `self.env` already exists with the correct number of blocks.

---

### 3. BaseGraphClutteredStorage2DTAMPSystem.__init__()
**File:** [clutteredstorage_system.py:259-278](src/tamp_improv/benchmarks/clutteredstorage_system.py#L259-L278)

```python
def __init__(self, planning_components, n_blocks=1, seed=None, render_mode=None, **kwargs):
    self._render_mode = render_mode
    # Only set n_blocks if not already set (to support cooperative inheritance)
    if not hasattr(self, 'n_blocks'):
        self.n_blocks = n_blocks
    super().__init__(planning_components, name="GraphClutteredStorage2DTAMPSystem",
                     seed=seed, render_mode=render_mode, **kwargs)
```

**Actions:**
- Checks if `n_blocks` already exists (it does, set to 3 in step 1)
- Skips setting `n_blocks` (preserves value of 3)
- Calls `super().__init__()` → Goes to next in MRO

---

### 4. BaseTAMPSystem.__init__()
**File:** [base.py:50-63](src/tamp_improv/benchmarks/base.py#L50-L63)

```python
def __init__(self, planning_components, name="TAMPSystem", seed=None, render_mode=None):
    self.name = name
    self.components = planning_components
    self.env = self._create_env()  # Line 60
    if seed is not None:
        self.env.reset(seed=seed)
    self._render_mode = render_mode
```

**Actions:**
- Line 60: **Creates `self.env = self._create_env()`** ← **Base environment created here!**
  - Uses `self.n_blocks = 3` (from step 1)
  - The `_create_env()` method is implemented in `BaseGraphClutteredStorage2DTAMPSystem`
- Line 61-62: Resets env with seed

---

## Where ImprovWrapper is Created

The `ImprovWrapper` is created at **Step 2, Line 120** ([base.py:120](src/tamp_improv/benchmarks/base.py#L120)):

```python
self.wrapped_env = self._create_wrapped_env(planning_components)
```

This calls the concrete implementation in [clutteredstorage_system.py:473-484](src/tamp_improv/benchmarks/clutteredstorage_system.py#L473-L484):

```python
def _create_wrapped_env(self, components):
    return ImprovWrapper(
        base_env=self.env,  # References the already-created base env with n_blocks=3
        perceiver=components.perceiver,
        step_penalty=-0.5,
        achievement_bonus=10.0,
    )
```

**Key Point:** By the time `_create_wrapped_env()` is called in Step 2:
- The `super().__init__()` chain (Steps 3-4) has **already completed**
- `self.env` has been created with the **correct** number of blocks (n_blocks=3)
- `self.n_blocks` was set at the very beginning in Step 1

## Execution Timeline

```
Time →

GraphClutteredStorage2DTAMPSystem.__init__ starts
  ↓ Sets self.n_blocks = 3
  ↓ Calls super().__init__()
  ↓
  ImprovisationalTAMPSystem.__init__ starts
    ↓ Calls super().__init__()
    ↓
    BaseGraphClutteredStorage2DTAMPSystem.__init__ starts
      ↓ Checks n_blocks (already set, skips)
      ↓ Calls super().__init__()
      ↓
      BaseTAMPSystem.__init__ starts
        ↓ Creates self.env via _create_env() (uses n_blocks=3)
        ↓ Resets self.env
      BaseTAMPSystem.__init__ completes
      ↓
    BaseGraphClutteredStorage2DTAMPSystem.__init__ completes
    ↓
    ← Back to ImprovisationalTAMPSystem.__init__
    ↓ Creates self.wrapped_env = ImprovWrapper(self.env)  ← ImprovWrapper wraps the correct env!
    ↓ Resets self.wrapped_env
  ImprovisationalTAMPSystem.__init__ completes
  ↓
GraphClutteredStorage2DTAMPSystem.__init__ completes
```

## Key Takeaways

1. **Only ONE environment is created** (in Step 4) with the correct configuration
2. **ImprovWrapper wraps the correct environment** (created in Step 2, after Step 4 completes)
3. **Attributes must be set BEFORE `super().__init__()`** to be available in nested calls
4. **Cooperative inheritance requires using `super()`** instead of explicit parent calls
5. **The `**kwargs` pattern** allows parent classes to accept and pass through parameters they don't directly use
