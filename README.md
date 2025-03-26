[![pre-commit](https://github.com/daniel-gallo/ssm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/daniel-gallo/ssm/actions/workflows/pre-commit.yml)
# Dev instructions
As linter / formatter, we can use [Ruff](https://docs.astral.sh/ruff/).

## Launching a sweep on TPUs
1. Edit `example-sweep.yaml` to configure the sweep, save the file as
`sweep.yaml`.
2. Run `wandb sweep sweep.yaml` to launch the sweep. The sweep id will be
   output to the terminal, as well as the URL for viewing the sweep online.
3. Two options for attaching nodes to the sweep:
     - Start a fresh node. Refer to the Google sheet to see what accelerator
       types are available, then use the following command:
       ```bash
       gcloud compute tpus tpu-vm create NAME_OF_TPU --zone=ZONE \
         --accelerator-type=ACCELERATOR_TYPE --version=tpu-ubuntu2204-base \
         [--preemptible] --metadata=wandb-sweep-id=SWEEP_ID \
         --metadata-from-file=startup-script=tpu-startup-script.sh
       ```
       Edit the sheet to show that you are using the node.
     - Attach an existing node. Firstly update the sweep id on the node, then
       run the startup script:
        1. These commands will update the sweep id and refresh the startup script
           on the existing node:
           ```bash
           gcloud compute tpus tpu-vm update NAME_OF_TPU --zone=ZONE \
             --update-metadata=wandb-sweep-id=SWEEP_ID
           gcloud compute tpus tpu-vm update NAME_OF_TPU --zone=ZONE \
             --metadata-from-file=wandb-sweep-id=SWEEP_ID
           ```
        3. Make sure there are no processes using the TPU on the node, then run
           ```bash
           gcloud compute tpus tpu-vm ssh NAME_OF_TPU --zone ZONE \
             --worker all \
             --command "sudo google_metadata_script_runner startup &> startup-log.txt &"
           ```

### Viewing the logs
In the fresh node case, run
```bash
sudo journalctl -u google-startup-scripts.service
```
on the node, you will need to scroll to the very bottom because there is a lot
of output _before_ the output of our startup script.

In the existing node case, the log is saved in `~/startup-log.txt`.


## Zed instructions
Add this to `~/.config/zed/settings.json`

```json
{
    "languages": {
        "Python": {
            "format_on_save": "on",
            "formatter": [
                {
                    "code_actions": {
                        "source.organizeImports.ruff": true,
                        "source.fixAll.ruff": true
                    }
                },
                {
                    "language_server": {
                        "name": "ruff"
                    }
                }
            ]
        }
    }
}

```
## Pre-commit hook
This ensures that your commits will stick to a pre-defined coding style.
1. Install `pre-commit` (using pip for example)
1. Run `pre-commit install`
1. (Optional) run `pre-commit run --all-files`

After installing the hook, some checks will be performed automatically before you commit your files.
