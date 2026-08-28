#!/bin/bash
# Submit the complete exp5 pipeline with SLURM dependencies.
#
# Usage:
#   export FLOWREC_PROJECT_ROOT=/path/to/flowrec
#   export EXP5_CONFIG=scripts/exp5_budgeted_cyclic/configs/beauty2014_full.yaml
#   export EXP5_ENV_SETUP=/path/to/env_setup.sh    # optional
#   bash scripts/exp5_budgeted_cyclic/slurm/submit_pipeline.sh
#
# Stage graph (spec Part IV):
#   00 -> 01 -> {02,03} x {validation,test} -> 04 -> 10(protocol regression)
#      -> 05(controller fit) -> 05(validation array) -> 06(select+freeze+gate)
#      -> 07(test array) -> 08(serving array) -> 09(aggregate+report)
# Test evaluation runs ONLY after stage 06 freezes the selection manifest.
set -euo pipefail
cd "$(dirname "$0")"

sub() {  # sub <dependency-or-empty> <env> <script> -> job id
  local dep="$1"; local env="$2"; local script="$3"
  local args=(--parsable)
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  [ -n "$env" ] && args+=(--export="ALL,$env")
  sbatch "${args[@]}" "$script" | cut -d';' -f1
}

j00=$(sub ""    ""                       00_validate_assets.sbatch)
j01=$(sub "$j00" ""                      01_build_manifests.sbatch)

j02v=$(sub "$j01" "EXP5_SPLIT=validation" 02_sasrec_array.sbatch)
j02t=$(sub "$j01" "EXP5_SPLIT=test"       02_sasrec_array.sbatch)
j03v=$(sub "$j01" "EXP5_SPLIT=validation" 03_tiger_array.sbatch)
j03t=$(sub "$j01" "EXP5_SPLIT=test"       03_tiger_array.sbatch)

j04v=$(sub "$j02v:$j03v" "EXP5_SPLIT=validation" 04_flow_cache_array.sbatch)
j04t=$(sub "$j02t:$j03t" "EXP5_SPLIT=test"       04_flow_cache_array.sbatch)

# Protocol regression (spec 5.4) gates the production grid.
j10=$(sub "$j03v" "" 10_protocol_regression.sbatch)

jctl=$(sub "$j04v:$j10" "" 05_controller_fit.sbatch)
j05=$(sub "$jctl" "EXP5_SPLIT=validation" 05_validation_array.sbatch)
j06=$(sub "$j05" "" 06_select.sbatch)
j07=$(sub "$j06:$j04t" "" 07_test_array.sbatch)
j08=$(sub "$j07" "EXP5_SPLIT=test" 08_serving_array.sbatch)
j09=$(sub "$j08" "" 09_report.sbatch)

echo "submitted: 00=$j00 01=$j01 02v=$j02v 02t=$j02t 03v=$j03v 03t=$j03t"
echo "           04v=$j04v 04t=$j04t 10=$j10 ctl=$jctl 05=$j05 06=$j06"
echo "           07=$j07 08=$j08 09=$j09"
