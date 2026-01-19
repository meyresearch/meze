#!/bin/bash

#SBATCH --job-name=ligand_11-g-opt
#SBATCH nodes=1
#SBATCH partition=short
#SBATCH ntasks-per-node=8
#SBATCH mem=24GB
#SBATCH account=

module load apps/gaussian

LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export GAUSS_SCRDIR=/Users/af25016/projects/meze/data//inputs/hybrid_model/protein/vim2/ligand_11/01_mcpb_parameterisation/scratch


g16 /Users/af25016/projects/meze/data//inputs/hybrid_model/protein/vim2/ligand_11/01_mcpb_parameterisation/vim2_ezaff_ligand_11_large_opt.com