#!/bin/bash
#SBATCH --job-name=aggcrerl_r2avg_grpo_lora2g_allgen
#SBATCH -p  mcml-hgx-h100-94x4 
#rrSBATCH -p  lrz-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2avg/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate trl
cd "$PROJECT_PATH"

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file ./examples/accelerate_configs/deepspeed_zero3_baseline2g.yaml \
    tmp_lora.py &>> ${JOBLOG}


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}