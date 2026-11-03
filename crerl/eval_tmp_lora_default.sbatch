#!/bin/bash
#SBATCH --job-name=eval_trl_grpobaseline_lora2g_default
#SBATCH -p  mcml-hgx-a100-80x4,mcml-hgx-h100-94x4 
#SBATCH --qos=mcml
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -t 12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/trl/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate eval
cd "$PROJECT_PATH"

export WANDB_PROJECT=lighteval
export WANDB_NAME=$SLURM_JOB_NAME
export WANDB_START_METHOD=thread

# 🔴 VERY IMPORTANT: avoid TorchDynamo bugs
export TORCH_COMPILE_DISABLE=1
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_NUM_PROCESSES=2

MODEL_PATH="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/trl/output/trl_grpobaseline_lora2g_5417061_merged"
TASK="custom|minervamath|0|0"

accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16  -m \
    lighteval accelerate "model_name=${MODEL_PATH},model_parallel=True,batch_size=16" "gsm8k" \
    --output-dir ./output_eval/$SLURM_JOB_NAME \
    --save-details --cache.force_refresh \
    --wandb &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}
