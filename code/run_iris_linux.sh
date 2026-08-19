#!/usr/bin/env bash
set -euo pipefail
./auto_dmc_ac \
  --partition-dir IrisPartition \
  --miner ./dmc_miner_unified \
  --output-dir iris_auto_results \
  --sigma 0.001 \
  --min-netconf 0.0 \
  --inner-folds 3 \
  --seed 20260715
