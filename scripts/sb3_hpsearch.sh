#!/usr/bin/env bash

# Submit a job array without a physical .sbatch file using config files a HERE document.
# https://en.wikipedia.org/wiki/Here_document
# https://slurm.schedmd.com/job_array.html
#
# Before submitting prepare a `queue` folder where each file corresponds to one config.
# Each file is called `array.<date>.<id>.yaml`. Files corresponding to succesful runs
# are deleted. If the run fails the config file is moved to an `error` folder.
#
# Variables and commands in the HERE document work like this:
# - ${RUNS_PATH}     is evaluated *now* and takes the value
#                    from the current shell (as defined below),
#                    it's useful to pass paths and thyperparameters
# - \${SLURM_JOB_ID} is evaluated when the job starts, therefore
#                    you can access variables set by slurm
# - $(date)          is evaluated *now* and takes the value
#                    from the current shell (as defined above)
# - \$(date)         is evaluated when the job starts, therefore
#                    you can run commands on the node

LOG_PATH="${HOME}/slurm_logs/dedo/"
SWEEP_ID=$1
USER=pyshi
SLURM_ARRAY_TASK_ID=0
# |shelob
sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${LOG_PATH}/%x-%J.log"
#SBATCH --error="${LOG_PATH}/%x-%J.log"
#SBATCH --mail-type=FAIL
#SBATCH --constrain="balrog|smaug|shelob|khazadum|belegost|shire|rivendell"
#SBATCH --mail-user="${USER}@kth.se"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --job-name=HPSearch-${SWEEP_ID}
# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "JOB NAME: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo ""
nvidia-smi
# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate DEDO
# Train and save the exit code of the python script
wandb agent eoai4globalchange/hp-search-dedo/${SWEEP_ID}
HERE