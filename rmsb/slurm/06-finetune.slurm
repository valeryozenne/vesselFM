#!/bin/bash
#SBATCH --job-name=vFM_finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hippolyte.salles@gmail.com

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.1

set -x

mkdir -p logs

cd /lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM:${PYTHONPATH:-}

python vesselfm/seg/finetune.py