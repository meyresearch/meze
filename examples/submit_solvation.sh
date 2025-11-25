#!/bin/bash

project_dir=$1

sbatch --error=../logs/solv_%x_%a.err \
       --output=../logs/solv_%x_%a.out \
       --array=1-16 \ #FIXME
       solvate.sh $project_dir 
