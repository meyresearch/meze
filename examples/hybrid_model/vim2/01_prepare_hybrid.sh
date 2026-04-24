#!/bin/bash

project_dir=$1
system_name=$2


for ligand_dir in ${project_dir}/inputs/hybrid_model/ligands/${system_name}/ligand_*
do
#    echo $ligand_dir
    filename=$(basename "$ligand_dir")
    ligand_name=${filename%.*}
    echo $ligand_name
    python prepare_hybrid_model.py $project_dir $system_name $ligand_name

done
