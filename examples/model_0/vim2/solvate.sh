#!/bin/bash
#SBATCH --nodes=1

project_dir=$1
system_name=$2
lig_i=$SLURM_ARRAY_TASK_ID

python solvate.py $project_dir $system_name "ligand_${lig_i}"
