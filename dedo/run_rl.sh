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
ENV_NAME=$1
MAX_STEPS=$2
USER=pyshi
SLURM_ARRAY_TASK_ID=0
# |shelob
sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${LOG_PATH}/%x%j.out"
#SBATCH --error="${LOG_PATH}/%x%j.err"
#SBATCH --mail-type=FAIL
#SBATCH --constrain="balrog|smaug|shelob|khazadum|belegost|shire|gondor|rivendell"
#SBATCH --mail-user="${USER}@kth.se"
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --job-name=${ENV_NAME}
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
python -m dedo.rl_demo --env=${ENV_NAME} --logdir=${HOME}/experiment_logs/dedo --cam_resolution 100 --num_play_runs=1 --max_episode_len=${MAX_STEPS}
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE