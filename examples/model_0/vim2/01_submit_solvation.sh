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
system_name=$2

sbatch --error=../logs/solvate_%x_%a.err \
       --output=../logs/solvate_%x_%a.out \
       --array="$arr_str" \ 
       solvate.sh $project_dir $system_name
