#!/bin/bash

POD_NAME="tpu-pod-0"

gcloud compute tpus tpu-vm scp scripts/pod-script.sh ${POD_NAME}:~ \
    --zone europe-west4-a \
    --project craystack-dev \
    --worker all

gcloud compute tpus tpu-vm ssh ${POD_NAME} \
    --zone europe-west4-a \
    --project craystack-dev \
    --worker all \
    --command "bash pod-script.sh"
