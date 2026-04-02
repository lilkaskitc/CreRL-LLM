#!/bin/bash
#SBATCH --job-name=eval_crerl_r2avg_grpo_lora4g_5467274
#SBATCH -p  mcml-hgx-a100-80x4,mcml-hgx-h100-94x4 
#rrSBATCH -p  lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH -t 06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2avg/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate eval
cd "$PROJECT_PATH"

export WANDB_PROJECT=lighteval
export WANDB_NAME=$SLURM_JOB_NAME
export WANDB_API_KEY="272f1b038ff8ae26ef5980d3a79e786f9631ee29"
export WANDB_START_METHOD=thread

# 🔴 VERY IMPORTANT: avoid TorchDynamo bugs
export TORCH_COMPILE_DISABLE=1
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_NUM_PROCESSES=1
export TRANSFORMERS_DISABLE_SDPA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2avg/output/crerl_r2avg_grpo_lora4g_5467274_merged"
TASKS="aime25m|0,aime24m|0,math_500m|0,minervamath|0,gsm8km|0,amc23|0,olympiadbench|0"
#TASKS="aime24m|0,gsm8km|0"

accelerate launch --num_processes=1 --mixed_precision=bf16  -m \
    lighteval accelerate \
    "model_name=${MODEL_PATH},model_parallel=True,batch_size=16,max_length=1536,generation_parameters={max_new_tokens:1024},cache_dir=None" \
     $TASKS \
    --custom-tasks ./trl/eval2.py \
    --output-dir ./output_eval/$SLURM_JOB_NAME \
    --save-details  \
    --wandb &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}