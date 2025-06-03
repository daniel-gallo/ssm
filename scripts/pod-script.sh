#!/bin/bash

# Get credentials
# Wandb
curl http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_id -H "Metadata-Flavor: Google" > ~/.netrc
chmod 600 ~/.netrc
# GitHub
mkdir -p ~/.ssh
curl http://metadata.google.internal/computeMetadata/v1/project/attributes/github_id_rsa -H "Metadata-Flavor: Google" > ~/.ssh/github_id_rsa
chmod 600 ~/.ssh/github_id_rsa
curl http://metadata.google.internal/computeMetadata/v1/project/attributes/github_id_rsa_pub -H "Metadata-Flavor: Google" > ~/.ssh/github_id_rsa.pub
chmod 644 ~/.ssh/github_id_rsa.pub
cat <<EOF > ~/.ssh/config
Host github.com
    IdentityFile ~/.ssh/github_id_rsa
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config
# HuggingFace
HF_TOKEN=$(curl http://metadata.google.internal/computeMetadata/v1/project/attributes/huggingface_token -H "Metadata-Flavor: Google")
export HF_TOKEN

# Get latest code / dependencies
if [ -d "ssm" ]; then
    # Not first run, just update
    cd ssm
    source venv/bin/activate
    git pull
    pip install -Ur requirements.txt
else
    # First run
    git clone git@github.com:daniel-gallo/ssm.git
    cd ssm
    export DEBIAN_FRONTEND=noninteractive
    export NEEDRESTART_SUSPEND=1
    export NEEDRESTART_MODE=a
    sudo --preserve-env=DEBIAN_FRONTEND,NEEDRESTART_SUSPEND,NEEDRESTART_MODE apt update -y
    sudo --preserve-env=DEBIAN_FRONTEND,NEEDRESTART_SUSPEND,NEEDRESTART_MODE apt upgrade -y
    sudo --preserve-env=DEBIAN_FRONTEND,NEEDRESTART_SUSPEND,NEEDRESTART_MODE apt install -y ffmpeg
    sudo --preserve-env=DEBIAN_FRONTEND,NEEDRESTART_SUSPEND,NEEDRESTART_MODE apt install -y python3.10-venv
    python -m venv venv
    source venv/bin/activate
    pip install jax[tpu]
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install tensorflow-cpu
    pip install -Ur requirements.txt
fi

# Run train.py detached
nohup python train.py patch-ar \
    --dataset=sc09 \
    --batch_size=32 \
    --num_epochs=500 \
    --pool_temporal="[2,2,1,1,1,1]" \
    --base_dim=128 \
    --dropout_rate=0.2 \
    --use_temporal_cnn=false \
    --conv_pooling=true \
    --model_structure="[
        ['rglru','rglru','rglru','rglru','rglru'],
        ['rglru','rglru','rglru','rglru','rglru'],
        ['rglru','rglru','rglru','rglru','rglru','rglru'],
        ['rglru','rglru','rglru','rglru','rglru','rglru'],
        ['rglru','rglru','rglru','rglru','rglru','rglru'],
        ['rglru','rglru','rglru','rglru','rglru','rglru','rglru','rglru'],
        ['rglru','rglru','rglru','rglru','rglru','rglru','rglru','rglru']
    ]" \
    --learning_rate=0.002 \
    --min_max_scaling=true &> nohup.out &
