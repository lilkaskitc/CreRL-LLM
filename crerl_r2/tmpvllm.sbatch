#!/bin/bash
#SBATCH --job-name=trl_tmpvllm
#SBATCH -p mcml-hgx-a100-80x4,mcml-dgx-a100-40x8,mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH -t 02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/trl/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"
# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate trl
cd "$PROJECT_PATH"

trl vllm-serve --model Qwen/Qwen2-0.5B-Instruct &>> ${JOBLOG} &
sleep 20

accelerate launch \
     --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
     tmpvllm.py --config ./examples/cli_configs/GRPObaseline.yaml --use_peft  &>> ${JOBLOG} &



echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}