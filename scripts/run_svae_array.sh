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
USER=pyshi
SLURM_ARRAY_TASK_ID=0
declare -A JOB_QUEUEUEUE
j=1
tt=17
for seed in {2,3,4,5}
do
for env in Sewing-v0 BGarments-v0 ButtonProc-v0      ### Outer for loop ###
do
for algo in VAE SVAE PRED DSA
do
echo $env ${algo} $seed
cat << HERE > "$HOME/queue/job${j}.sh"
#!/usr/bin/env bash
ENV_NAME=$env
ALGO=$algo
SEED=$seed
HERE
((j=j+1))
done
done
done

echo "${JOB_QUEUEUEUE[$tt,1]}"


LOG_PATH="${HOME}/slurm_logs/dedo/"
USER=pyshi
SLURM_MAX_TASKS=30
SLURM_ARRAY_TASK_TOTAL=48
# |shelob
sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${LOG_PATH}/%x_%a.out"
#SBATCH --error="${LOG_PATH}/%x_%a.err"
#SBATCH --mail-type=FAIL
#SBATCH --constrain="balrog|smaug|shelob|khazadum|belegost|shire|rivendell"
#SBATCH --mail-user="pyshi@kth.se"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --job-name=DEADLINE_TMR!!!!
#SBATCH --array=1-${SLURM_ARRAY_TASK_TOTAL}%${SLURM_MAX_TASKS}
# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "JOB NAME: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo ""

source $HOME/queue/job\${SLURM_ARRAY_TASK_ID}.sh
echo "$HOME/queue/job\${SLURM_ARRAY_TASK_ID}.sh"
echo "a ${SLURM_ARRAY_TASK_ID}"
echo "b \${SLURM_ARRAY_TASK_ID}"
echo \${ENV_NAME}
echo \$SEED
echo \$ALGO
echo "d \${SLURM_ARRAY_TASK_ID}"

nvidia-smi
# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate DEDO
# Train and save the exit code of the python script
python -m dedo.svae_demo --cam_resolution=256 --total_env_steps=300000 --num_envs=8 --logdir=~/experiment_logs/dedo --unsup_algo \${ALGO} --use_wandb --env \${ENV_NAME} --seed=\${SEED} --disable_logging_video
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE