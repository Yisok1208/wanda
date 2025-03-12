#!/bin/bash
#SBATCH --job-name=llama_3.2_1B_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=job_output_llama_2_7B_hf_sparse_0.5%j.log
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
          --model meta-llama/Llama-2-7b-chat-hf \
          --prune_method sparsegpt \
          --sparsity_ratio 0.4 \
          --sparsity_type unstructured \
          --eval_zero_shot \
          --save "/mnt/parscratch/users/aca22yn/wanda/results/llama-2-7b-chat-hf_sgpt_0.4 /" \
          --save_model "/mnt/parscratch/users/aca22yn/wanda/pruned/sgpt/llama-2-7b-chat-hf_sgpt_0.4 /" \