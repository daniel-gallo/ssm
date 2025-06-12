#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/parallel.sh"

tpus=("tpu" "tpu1" "tpu3" "tpu-pod-0" "tpu-pod-1")

check_tpu() {
    local tpu="$1"
    echo "== Checking $tpu =="
    ssh "${tpu}" "cat ssm/nohup.out | grep epoch | tail -n 1"
    ssh "${tpu}" "ps aux | grep [t]rain.py " | awk '{ for(i=11;i<=NF;++i) printf $i " " }'
    echo
    echo
}

run_parallel check_tpu "${tpus[@]}"
