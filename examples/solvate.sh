#!/bin/bash
#SBATCH --nodes=1

project_dir=$1
lig_i=$SLURM_ARRAY_TASK_ID

python solvate.py $project_dir "ligand_${lig_i}"
