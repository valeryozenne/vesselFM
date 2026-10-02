#!/bin/bash
#SBATCH --job-name=vesselfm_seg_h100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
##SBATCH --qos=qos_gpu_h100-dev

set -euo pipefail

mkdir -p logs

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.1

cd /lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM

source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM:${PYTHONPATH:-}

MASTER_PORT=29500
MASTER_ADDR=127.0.0.1

torchrun --nproc_per_node=4 --nnodes=1 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m vesselfm.seg.train \
  devices='[0,1,2,3]' \
  trainer.lightning_trainer.accelerator=gpu \
  trainer.lightning_trainer.strategy=ddp \
  trainer.lightning_trainer.num_nodes=1 \
  trainer.lightning_trainer.log_every_n_steps=10 \
  trainer.lightning_trainer.precision=16-mixed \
  batch_size=8 \
  lr=1e-4 \
  seed=404