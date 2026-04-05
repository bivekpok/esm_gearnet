#!/bin/bash
#
# Submit:  sbatch run_opm_esm_2.sh
# Do NOT run directly (./run_opm_esm_2.sh) — #SBATCH lines are ignored outside Slurm.
#
#SBATCH --job-name="esmopm2"
#SBATCH --output="/work/hdd/bdja/bpokhrel/esm_new2/log/esmopm2.out.%j.%N.out"
#SBATCH --error="/work/hdd/bdja/bpokhrel/esm_new2/log/esmopm2.err.%j.%N.err"
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdja-delta-gpu    
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  
#SBATCH --gpus-per-node=2
#SBATCH --mail-user='bivekpok@udel.edu'
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -t 4:00:00

# 1. Clear loaded modules
module purge

# 2. Fix the NumExpr warning
export NUMEXPR_MAX_THREADS=8

# 3. Activate environment
source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate /u/bpokhrel/miniconda3/envs/esm

cd /work/hdd/bdja/bpokhrel/esm_new2/

# 4. Prevent W&B from filling up your limited home directory quota
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp

# 5. Help PyTorch deal with memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 6. DDP training — torchrun spawns 2 workers (one per GPU)
#    Change --outer / --inner to run different folds
echo "Launching DDP training on 2 GPUs..."
torchrun --standalone --nproc_per_node=2 \
    train.py --outer 1 --inner 1
