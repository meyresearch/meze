#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load apps/amber

LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

project_dir=$1
lig_i=$SLURM_ARRAY_TASK_ID
repeat=$2
system_name="l1" 

python heat_meze.py $project_dir $system_name "ligand_${lig_i}" $repeat
