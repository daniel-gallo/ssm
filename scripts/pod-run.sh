#!/bin/bash

POD_NAME="$1"
SCRIPT_NAME="$2"

if [[ -z "$POD_NAME" || -z "$SCRIPT_NAME" ]]; then
    echo "Usage: $0 <POD_NAME> <SCRIPT_NAME>"
    echo Example: $0 tpu-pod-0 pod-train.sh
    exit 1
fi

gcloud compute tpus tpu-vm scp scripts/${SCRIPT_NAME} ${POD_NAME}:~ \
    --zone europe-west4-a \
    --project craystack-dev \
    --worker all

gcloud compute tpus tpu-vm ssh ${POD_NAME} \
    --zone europe-west4-a \
    --project craystack-dev \
    --worker all \
    --command "nohup bash ${SCRIPT_NAME}"
