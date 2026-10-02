#!/bin/bash
#SBATCH --job-name=sample_dflow
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=100:00:00
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
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM:${PYTHONPATH:-}


torchrun --standalone --nproc_per_node=4 vesselfm/d_flow/sample.py \
    --ckpt_path runs/fm/model-15.pt \
    --mask_folder data/d_drand/d_drand \
    --out_folder data/d_flow \
    --batchsize 4 \
    --num_samples 10000 \
    --class_cond \
    --num_classes 15 \
    --start_id 508
    # --no_npy \
    # --nifti
    # --overview \
