#!/bin/bash
#SBATCH --job-name=mistral_7B_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=mistral_7B_full_%j.log
#SBATCH --mail-type=ALL

export SLURM_EXPORT_ENV=ALL

# Load any required modules
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load Anaconda3/2024.02-1

#Activate conda
source activate prune_llm

# Set Hugging Face cache environment variables
export HF_HOME="/mnt/parscratch/users/aca22yn/cache"
export HF_DATASETS_CACHE="/mnt/parscratch/users/aca22yn/cache/datasets"
export TRANSFORMERS_CACHE="/mnt/parscratch/users/aca22yn/cache/transformers"

# Set the HF_TOKEN environment variable by reading the file
export HF_TOKEN=$(cat /mnt/parscratch/users/aca22yn/wanda/token/hf_token.txt)

srun python /mnt/parscratch/users/aca22yn/yik/wanda/main_opt.py \
          --model mistralai/Mistral-7B-v0.3 \
          --sparsity_ratio 0 \
          --sparsity_type unstructured \
          --prune_method wanda \
          --eval_zero_shot
          --save "/mnt/parscratch/users/aca22yn/yik/wanda/results/mistral_7b_full/" \
          --save_model "/mnt/parscratch/users/aca22yn/yik/wanda/pruned/sgpt//mistral_7b_full/"
