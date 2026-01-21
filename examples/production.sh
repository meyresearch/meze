#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=
#SBATCH --job-name=prod0
#SBATCH --time=02-00:00:00
#SBATCH --account=

module load apps/amber

LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

project_dir=$1
lig_i=$SLURM_ARRAY_TASK_ID
repeat=$2

python production.py $project_dir "ligand_${lig_i}" $repeat
