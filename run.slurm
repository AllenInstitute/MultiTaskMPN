#!/bin/bash

#SBATCH --job-name=delaygo   # Job name
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought             
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task (process)
#SBATCH --mem=10G                        # Memory per node (e.g., 4 GB)
#SBATCH --time=12:00:00                # Time limit (hh:mm:ss)
#SBATCH --gpus=1


#SBATCH --export=all
#SBATCH --output=output/delaygo.txt
#SBATCH --error=output/delaygo.txt

python -u test.py
