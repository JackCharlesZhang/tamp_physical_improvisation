#!/bin/bash

# Define configurations and their overrides
# Each entry in CONFIGS_TO_RUN is a space-separated string:
# "config_name override1=value1 override2=value2 ..."
CONFIGS_TO_RUN=(
  "gridworld_fixed heuristic.num_epochs_per_round=100"
)

# Loop through each config entry and submit a SLURM job
for config_entry in "${CONFIGS_TO_RUN[@]}"; do
  # Split the entry into config_name and overrides
  read -r config_name CONFIG_OVERRIDES <<< "$config_entry"

  echo "Submitting SLURM job for config: $config_name with overrides: ${CONFIG_OVERRIDES}"

  # Set the CONFIG_OVERRIDES environment variable for run_slap_v2.sh
  export CONFIG_OVERRIDES="${CONFIG_OVERRIDES}"

  # Submit the job
  sbatch scripts/run_slap_v2.sh --config-name "$config_name"

  # Unset CONFIG_OVERRIDES to prevent it from affecting subsequent sbatch calls if not explicitly set
  unset CONFIG_OVERRIDES

  sleep 1 # Add a small delay to avoid overwhelming the SLURM scheduler
done

echo "All SLURM jobs submitted."
