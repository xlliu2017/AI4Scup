#!/bin/bash -l
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:a100:1   
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Hnet
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --mem=128G


eval "$(conda shell.bash hook)"

# Change to the name to your conda env.

conda activate MgNO 
python /ibex/user/liux0t/AI4S-cupv2/train2.py -c '/ibex/user/liux0t/AI4S-cupv2/config/MgNO.yaml'