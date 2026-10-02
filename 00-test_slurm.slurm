#!/bin/bash
#SBATCH --job-name=test_email
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --time=01:00:00
#SBATCH --partition=visu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hippolyte.salles@gmail.com

set -euo pipefail

module purge
module load python

python - <<'PY'
import numpy as np
data = np.random.rand(1000000)
mean = np.mean(data)
std = np.std(data)
print(f"Mean: {mean}, Standard Deviation: {std}")
PY

