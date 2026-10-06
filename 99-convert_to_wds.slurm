#!/bin/bash
##srun --pty --ntasks=1 --cpus-per-task=4 --hint=nomultithread --partition=archive bash
#SBATCH --job-name=archive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=archive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hippolyte.salles@gmail.com

set -x

module load python

cd $SCRATCH/vesselFM
source venv-wds/bin/activate

python convert_to_wds.py --drand --dir $WORK_SHARED/vesselFM/data