# Della-GPU Environment Setup

## Conda Environment

**Environment Name**: `tamp-improv`

**Location**: `/home/jz4267/.conda/envs/tamp-improv`

**Python Version**: 3.11

## Activation

```bash
ssh della-gpu
conda activate tamp-improv
```

## Verify Environment

After activation, verify the setup:

```bash
# Check Python version
python --version  # Should be Python 3.11.x

# Check key packages
pip list | grep prbench
pip list | grep relational-structs
pip list | grep bilevel-planning
```

## Project Location on Della-GPU

**Code Directory**: `/scratch/gpfs/jz4267/tamp_physical_improvisation`

(Update this path if different)

## Running SLAP Training

```bash
# Activate environment
conda activate tamp-improv

# Navigate to project
cd /scratch/gpfs/jz4267/tamp_physical_improvisation

# Run SLAP training for DynObstruction2D
python experiments/slap_train.py --system DynObstruction2DTAMPSystem
```

## Other Available Environments (for reference)

- `myenv` - /home/jz4267/.conda/envs/myenv
- `py312` - /home/jz4267/.conda/envs/py312
- `slap` - /home/jz4267/.conda/envs/slap
- `base` - /usr/licensed/anaconda3/2024.10

**Use `tamp-improv` for all SLAP/DynObstruction2D work.**
