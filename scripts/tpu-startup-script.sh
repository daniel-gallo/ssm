#!/usr/bin/bash

# SET HOSTNAME
accty=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type -H "Metadata-Flavor: Google")
instanceid=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance-id -H "Metadata-Flavor: Google")
if [[ $accty = v2-8 || $accty = v3-8 || $accty = v4-8 ]]
then
  hostname=$instanceid
else
  workerid=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number -H "Metadata-Flavor: Google")
  hostname=$instanceid-$workerid
fi
hostnamectl set-hostname $hostname

# INSTALL FFMPEG
apt install ffmpeg

# SETUP GITHUB ACCESS
mkdir ~/.ssh
curl http://metadata.google.internal/computeMetadata/v1/project/attributes/github_id_rsa -H "Metadata-Flavor: Google" > ~/.ssh/github_id_rsa
chmod 600 ~/.ssh/github_id_rsa
curl http://metadata.google.internal/computeMetadata/v1/project/attributes/github_id_rsa_pub -H "Metadata-Flavor: Google" > ~/.ssh/github_id_rsa.pub
echo "Host github.com
    IdentityFile ~/.ssh/github_id_rsa
    StrictHostKeyChecking no" > ~/.ssh/config

# SETUP WANDB ACCESS
curl http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_id -H "Metadata-Flavor: Google" > ~/.netrc

# INSTALL JAX, CPU PYTORCH AND CPU TF
pip install -U jax[tpu]
pip install -U torch --index-url https://download.pytorch.org/whl/cpu
pip install -U tensorflow-cpu

# CLONE REPO, INSTALL DEPENDENCIES
rm -rf ssm
git clone git@github.com:daniel-gallo/ssm.git
cd ssm
pip install -Ur requirements.txt

# SETUP HUGGINGFACE ACCESS
huggingface_token=$(curl http://metadata.google.internal/computeMetadata/v1/project/attributes/huggingface_token -H "Metadata-Flavor: Google")
huggingface-cli login --token $huggingface_token

# LAUNCH SWEEP AGENT
wandb init --entity j-towns-org --project ssm
sweepid=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/wandb-sweep-id -H "Metadata-Flavor: Google")
# The TPU_... variables are necessary on a pod slice to say that we're not running a multi-host job
TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 wandb agent $sweepid
