#!/bin/bash

mapfile -t files < ligands.txt   

numbers=()
for f in "${files[@]}"; do
  numbers+=( "${f#ligand_}" )  
done

#echo "${numbers[@]}"

arr_str=$(IFS=,; echo "${numbers[*]}")

module load apps/amber

LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

project_dir=$1

sbatch --error=../logs/solv_%x_%a.err \
       --output=../logs/solv_%x_%a.out \
       --array="$arr_str" \
       scripts/solvate.sh $project_dir 
