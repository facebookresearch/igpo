#!/bin/bash
#SBATCH --job-name=igpo
#SBATCH --output=igpo_logs/igpo_%j.out
#SBATCH --error=igpo_logs/igpo_%j.err
#SBATCH --nodes=1
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --mem=800G

export WANDB_API_KEY=b0015ef35ecd6407c6088354f1e7b557db4d3e1c
# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate igpo

# Ensure .local/bin is in PATH for ninja
export PATH=$HOME/.local/bin:$PATH

# Create logs directory if it doesn't exist
mkdir -p igpo_logs
bash run_igpo.sh