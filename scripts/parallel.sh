#!/bin/bash

# run_parallel accepts a function name and a list of parameters.
# It will run the function on each parameter in parallel.
run_parallel() {
    local task_func="$1"
    shift
    local -a temp_files=()

    for arg in "$@"; do
        local tmpfile=$(mktemp)
        temp_files+=("$tmpfile")

        (
            "$task_func" "$arg"
        ) > "$tmpfile" 2>&1 &
    done

    wait

    for tmpfile in "${temp_files[@]}"; do
        cat "$tmpfile"
        rm "$tmpfile"
    done
}
