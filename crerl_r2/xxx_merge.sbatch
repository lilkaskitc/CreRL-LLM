#!/bin/bash
#SBATCH --job-name=merge
#rrSBATCH -p  mcml-hgx-h100-94x4 
#SBATCH -p  lrz-hgx-h100-94x4
#rrSBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate trl
cd "$PROJECT_PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python merge_lora.py &>> ${JOBLOG}


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}