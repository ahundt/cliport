#!/bin/bash
#SBATCH --job-name=unit-test-networks
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wagnew3@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate cliport
nvidia-smi
cd /gscratch/prl/wagnew3/cliport/
echo "-----------"
CLIPORT_ROOT=/gscratch/prl/wagnew3/cliport/ python cliport/eval.py $1

