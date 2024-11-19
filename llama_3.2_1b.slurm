#!/bin/bash
#SBATCH --job-name=llama_3.2_1B_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=job_output_llama_3.2_1B_%j.log
#SBATCH --mail-user=yng9@sheffield.ac.uk
#SBATCH --mail-type=ALL

export SLURM_EXPORT_ENV=ALL

# Load any required modules
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Anaconda3/2024.02-1

#Activate conda
source activate prune_llm

# Set Hugging Face cache environment variables
export HF_HOME="/mnt/parscratch/users/aca22yn/cache"
export HF_DATASETS_CACHE="/mnt/parscratch/users/aca22yn/cache/datasets"
export TRANSFORMERS_CACHE="/mnt/parscratch/users/aca22yn/cache/transformers"

# Set the HF_TOKEN environment variable by reading the file
export HF_TOKEN=$(cat /mnt/parscratch/users/aca22yn/wanda/token/hf_token.txt)

srun python /mnt/parscratch/users/aca22yn/wanda/main.py \
          --model baffo32/decapoda-research-llama-7B-hf \
          --prune_method wanda \
          --sparsity_ratio 0.5 \
          --sparsity_type unstructured \
          --save "/mnt/parscratch/users/aca22yn/wanda/results/llama_3.2_1B/"

