#!/bin/bash
#SBATCH --job-name=eval_all_merged_5491876
#rrSBATCH -p  mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH -p  lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#rrSBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH -t 10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate eval
cd "$PROJECT_PATH"

# ============================================
# Configuration - Modify these as needed
# ============================================

# Merged checkpoints directory (output from merge script)
MERGED_DIR="${1:-/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2/output_merged/aggcrerl_r2_grpo_lora4g_baseline_5491876}"

# WandB configuration
export WANDB_PROJECT=lighteval
export WANDB_NAME="eval_merged_$(basename $MERGED_DIR)"
export WANDB_API_KEY="272f1b038ff8ae26ef5980d3a79e786f9631ee29"
export WANDB_START_METHOD=thread

# Evaluate every N steps (e.g., every 10 steps)
EVAL_EVERY="${2:-10}"

# Tasks to evaluate
TASKS="aime25m|0,aime24m|0,math_500m|0,minervamath|0,gsm8km|0,amc23|0,olympiadbench|0"
#TASKS="gsm8km|0"  # For quick testing

# Output directory
OUTPUT_DIR="./output_eval/$(basename $MERGED_DIR)"

# ============================================
# Evaluate all merged checkpoints
# ============================================

echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluating merged checkpoints from: $MERGED_DIR" &>> ${JOBLOG}
echo "$(date '+%Y-%m-%d %H:%M:%S') Eval every $EVAL_EVERY steps" &>> ${JOBLOG}

python eval_all_merged_checkpoints.py \
    --merged_dir "$MERGED_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_NAME" \
    --wandb_api_key "$WANDB_API_KEY" \
    --eval_every "$EVAL_EVERY" \
    --tasks "$TASKS" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --max_length 1536 \
    --max_new_tokens 1024 \
    --skip_existing \
    &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}
