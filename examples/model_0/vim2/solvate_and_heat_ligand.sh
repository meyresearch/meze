#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_short
#SBATCH --job-name=heat0
#SBATCH --time=2:00:00
#SBATCH --account=chem037584

module load apps/amber

LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

project_dir=$1
lig_i=$SLURM_ARRAY_TASK_ID
repeat=$2

python solvate_and_heat_ligand.py $project_dir "ligand_${lig_i}" $repeat
