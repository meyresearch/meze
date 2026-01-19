#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --job-name=g-opt
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24GB
#SBATCH --account=ACCOUNT

module load apps/gaussian

LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export GAUSS_SCRDIR=SCRATCHDIR

g16 system_large_opt.com