#!/bin/bash
#SBATCH --job-name=llama_3.1_8b_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --output=job_output_llama_3.1_8b_sparse_0.5%j.log
#SBATCH --mail-type=ALL

export SLURM_EXPORT_ENV=ALL

# Load any required modules
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load Anaconda3/2024.02-1

# Set Hugging Face cache environment variables
export HF_HOME="/mnt/parscratch/users/aca22yn/cache"
export HF_DATASETS_CACHE="/mnt/parscratch/users/aca22yn/cache/datasets"
export TRANSFORMERS_CACHE="/mnt/parscratch/users/aca22yn/cache/transformers"

# Set the HF_TOKEN environment variable by reading the file
export HF_TOKEN=$(cat /mnt/parscratch/users/aca22yn/wanda/token/hf_token.txt)

/mnt/parscratch/users/aca22yn/anaconda/envs/dissertation_llm/bin/python /mnt/parscratch/users/aca22yn/wanda2/wanda/main.py \
          --model "meta-llama/Llama-3.2-3B" \
          --nsmamples 4
          --prune_method sparsegpt \
          --sparsity_ratio 0.5 \
          --sparsity_type unstructured \
          --eval_zero_shot \
          --save "/mnt/parscratch/users/aca22yn/wanda2/wanda/results/llama_3.2_3b_sgpt_0.5/" \
          --save_model "/mnt/parscratch/users/aca22yn/wanda2/wanda/pruned/sgpt/llama_3.2_3b_sgpt_0.5/" \
