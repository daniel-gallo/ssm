#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/parallel.sh"

tpus=("tpu" "tpu1" "tpu3" "tpu-pod-0" "tpu-pod-1")
checkpoints="b577bff7|ec36b4d8"

find_checkpoints()  {
    local tpu="$1"
    echo "== Checking $tpu =="
    ssh "${tpu}" "find ssm/checkpoints | grep -E '${checkpoints}'"
    echo
    echo
}


run_parallel find_checkpoints "${tpus[@]}"
