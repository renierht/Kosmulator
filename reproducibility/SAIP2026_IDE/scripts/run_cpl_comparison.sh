#!/usr/bin/env bash
set -eo pipefail
source /mnt/c/Users/Marcel/PhD/Kosmulator_test/Clik/code/plc_3.0/plc-3.1/bin/clik_profile.sh
export PATH=/home/marcel/miniconda3/envs/Kosmulator_test/bin:$PATH
export CONDA_PREFIX=/home/marcel/miniconda3/envs/Kosmulator_test
export MPLBACKEND=Agg
cd "/mnt/c/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_followup"
exec python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py --only-model cpl --run-name CPL_comparison_v1 --num-cores 8 --burn 1000 --max-steps 100000 --init-jitter-frac 0.01 "$@"
