#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/parallel.sh"

tpus=("tpu" "tpu1" "tpu3" "tpu-pod-0" "tpu-pod-1" "tpu-pod-2")
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_regex>"
    exit 1
fi
checkpoints="$1"

find_checkpoints()  {
    local tpu="$1"
    echo "== Checking $tpu =="
    ssh "${tpu}" "find ssm/checkpoints | grep -E '${checkpoints}'"
    echo
    echo
}


run_parallel find_checkpoints "${tpus[@]}"
