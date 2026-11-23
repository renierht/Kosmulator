#!/bin/bash
set -eo pipefail
source /home/marcel/miniconda3/etc/profile.d/conda.sh
conda activate Kosmulator_test
source /mnt/c/Users/Marcel/PhD/Kosmulator_test/Clik/code/plc_3.0/plc-3.1/bin/clik_profile.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
study_root='/mnt/c/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_IDE_main_integration'
python "$study_root/validation/nonlinear_same_point/evaluate.py" '/mnt/c/Users/Marcel/PhD/Kosmulator_test_Class/Kosmulator_referee_clean' "$study_root/validation/nonlinear_same_point/paper_v2.json" > "$study_root/validation/nonlinear_same_point/paper_v2.log" 2>&1
python "$study_root/validation/nonlinear_same_point/evaluate.py" "$study_root" "$study_root/validation/nonlinear_same_point/current.json" > "$study_root/validation/nonlinear_same_point/current.log" 2>&1
