#!/bin/bash
#SBATCH --job-name=vesselfm_drand
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
##SBATCH --qos=qos_prepost

set -euo pipefail

mkdir -p logs

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.1

source /lustre/fsn1/projects/rech/fxj/uwv71fy/cats_vs_dogs/vesselFM/venv/bin/activate

export PYTHONPATH=/lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM:${PYTHONPATH:-}

cd /lustre/fsn1/projects/rech/fxj/uwv71fy/vesselFM

# None geom, perlin noise, 20000 samples
python vesselfm/d_drand/gen_bg.py

# voronoi, perlin noise, 10000 samples
python vesselfm/d_drand/gen_bg.py --mode_bg_geom voronoi --mode_bg_noise perlin --num_imgs 10000

# spheres, perlin noise, 10000 samples
python vesselfm/d_drand/gen_bg.py --mode_bg_geom spheres --mode_bg_noise perlin --first_idx 7088 --num_imgs 10000

# None geom, plain noise, 10000 samples
python vesselfm/d_drand/gen_bg.py --mode_bg_noise plain --num_imgs 10000

# merge all the generated backgrounds with forgrounds
python vesselfm/d_drand/gen_data.py
