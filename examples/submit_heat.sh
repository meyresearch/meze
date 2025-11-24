#!/bin/bash

project_dir=$1

for i in 1 2 3
do
	sbatch --error=../logs/heat_%x_%a.err \
       	       --output=../logs/heat_%x_%a.out \
       	       --array=1-16 \ #FIXEM
       	       heat.sh $project_dir $i
done
