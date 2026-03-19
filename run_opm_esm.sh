#!/bin/bash
#SBATCH --job-name="esmopm"
#SBATCH --output="/work/hdd/bdja/bpokhrel/esm_new2/log/esmopm.out.%j.%N.out"
#SBATCH --error="/work/hdd/bdja/bpokhrel/esm_new2/log/esmopm.err.%j.%N.err"
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdja-delta-gpu    
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # , 2
#SBATCH --cpus-per-task=16  
#SBATCH --gpus-per-node=1     #, 2    # always equal to "ntasks-per-node"
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest  
#SBATCH --mail-user='bivekpok@udel.edu'
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -t 48:00:00

# 1. Clear loaded modules
module purge

# 2. Fix the NumExpr warning you were getting
export NUMEXPR_MAX_THREADS=8

# 3. Activate your environment
# Note: Adjust this line depending on if you use conda or python venv
source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate /u/bpokhrel/miniconda3/envs/esm

cd /work/hdd/bdja/bpokhrel/esm_new2/
# 2. Prevent W&B from filling up your limited home directory quota
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp

echo "Starting W&B agent..."
wandb agent udel/OPMesm-protein-localization/2f46fn6w