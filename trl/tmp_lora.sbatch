#!/bin/bash
#SBATCH --job-name=trl_grpobaseline_lora2g
#rrSBATCH -p  mcml-hgx-a100-80x4,mcml-hgx-h100-94x4 
#SBATCH -p  lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#rrSBATCH --qos=mcml
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
mamba activate trl
cd "$PROJECT_PATH"
accelerate launch --config_file ./examples/accelerate_configs/deepspeed_zero3_baseline2g.yaml \
    tmp_lora.py &>> ${JOBLOG}


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}