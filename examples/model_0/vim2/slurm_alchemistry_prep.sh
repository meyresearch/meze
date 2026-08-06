#!/bin/bash
#SBATCH --job-name=alchemistry
#SBATCH ...

PROJECT_DIR=$1
SYSTEM_NAME=$2
NETWORK_FILE=$3

LINE=$(awk -v idx=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==idx' "${NETWORK_FILE}")
LIGAND_1=$(echo $LINE | cut -d',' -f1)
LIGAND_2=$(echo $LINE | cut -d',' -f2)

python prepare_alchemistry.py "${PROJECT_DIR}" "${SYSTEM_NAME}" "${LIGAND_1}" "${LIGAND_2}"
