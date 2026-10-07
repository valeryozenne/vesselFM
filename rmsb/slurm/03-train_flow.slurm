#!/bin/bash
#SBATCH --job-name=vesselfm_flow
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=40:00:00
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hippolyte.salles@gmail.com

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.1

set -x

mkdir -p logs

cd /lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM

# Install package in editable mode for the user (only once per environment)
source venv/bin/activate
torchvision may already be provided by the module; try importing and install only if missing
python - <<'PY'
try:
    import torchvision
    print('torchvision available')
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'torchvision'])
PY

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM:${PYTHONPATH:-}

# Rendezvous port
MASTER_PORT=29500
MASTER_ADDR=127.0.0.1

# Launch training with torchrun (one process per GPU)
# Adjust --nproc_per_node if you change number of requested GPUs
torchrun --nproc_per_node=4 --nnodes=1 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  vesselfm/d_flow/diffusion.py