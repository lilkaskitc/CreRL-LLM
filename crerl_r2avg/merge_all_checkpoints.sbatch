#!/bin/bash
#SBATCH --job-name=merge_all_54996617
#SBATCH -p  mcml-hgx-h100-94x4
#rrSBATCH -p  lrz-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 02:00:00
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# Configuration - Modify these as needed
# ============================================

# Training output directory containing LoRA checkpoints
TRAIN_DIR="${1:-/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2avg/output/aggcrerl_r2avg_grpo_lora4g_5496617}"

# Base model for merging
BASE_MODEL="${2:-Qwen/Qwen2.5-Math-7B}"

# Merge every N steps (e.g., every 10 steps)
EVAL_EVERY="${3:-10}"

# Output directory for merged checkpoints
MERGED_DIR="./output_merged"

# ============================================
# Merge all checkpoints
# ============================================

echo "$(date '+%Y-%m-%d %H:%M:%S') Merging checkpoints from: $TRAIN_DIR" &>> ${JOBLOG}
echo "$(date '+%Y-%m-%d %H:%M:%S') Base model: $BASE_MODEL" &>> ${JOBLOG}
echo "$(date '+%Y-%m-%d %H:%M:%S') Merge every $EVAL_EVERY steps" &>> ${JOBLOG}

python merge_all_checkpoints.py \
    --train_dir "$TRAIN_DIR" \
    --base_model "$BASE_MODEL" \
    --merged_dir "$MERGED_DIR" \
    --eval_every "$EVAL_EVERY" \
    --skip_existing \
    &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}
