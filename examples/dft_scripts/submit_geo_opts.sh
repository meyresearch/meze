#!/bin/bash

project_dir=$1
system_name=$2

for ligand_dir in ${project_dir}/ligand_*
do
    ligand_name=$(basename "$ligand_dir")
	echo $ligand_name

    param_dir=${ligand_dir}/01_mcpb_parameterisation/
    cd $param_dir
    sbatch slurm_g_opt.sh

done
