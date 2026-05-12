#!/bin/bash
# Usage: bash submit_alchemistry.sh <project_dir> <system_name>
PROJECT_DIR=$1
SYSTEM_NAME=$2

SOFRA_JSON=$3
# DEBUGGING:
SOFRA_JSON="${PROJECT_DIR}/inputs/model_0/protein/${SYSTEM_NAME}/model_0_sofra.json"

NETWORK_FILE=$(python -c "import json; print(json.load(open('${SOFRA_JSON}'))['network_file'])")

N=$(tail -n +2 "${NETWORK_FILE}" | wc -l | tr -d ' ')

sbatch --array=0-$((N-1)) slurm_alchemistry_prep.sh \
    "${PROJECT_DIR}" "${SYSTEM_NAME}" "${NETWORK_FILE}"
