#!/bin/bash

project_dir=$1
system_name=$2


for ligand_dir in ${project_dir}/ligand_*
do
    ligand_name=$(basename "$ligand_dir")
	echo $ligand_name
    python prepare_hybrid_model.py $project_dir $system_name $ligand_name

done
