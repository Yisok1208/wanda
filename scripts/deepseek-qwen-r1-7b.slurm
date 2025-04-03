#!/bin/bash
#SBATCH --job-name=deepseek_7b_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=job_output_deepseek_7b_sparse_0.5%j.log
#SBATCH --mail-type=ALL

export SLURM_EXPORT_ENV=ALL

# Load any required modules
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load Anaconda3/2024.02-1

#Activate conda
source /mnt/parscratch/users/aca22yn/anaconda/etc/profile.d/conda.sh
conda activate dissertation_llm

# Set Hugging Face cache environment variables
export HF_HOME="/mnt/parscratch/users/aca22yn/cache"
export HF_DATASETS_CACHE="/mnt/parscratch/users/aca22yn/cache/datasets"
export TRANSFORMERS_CACHE="/mnt/parscratch/users/aca22yn/cache/transformers"

# Set the HF_TOKEN environment variable by reading the file
export HF_TOKEN=$(cat /mnt/parscratch/users/aca22yn/wanda/token/hf_token.txt)

srun python /mnt/parscratch/users/aca22yn/wanda2/wanda/main.py \
          --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
          --prune_method sparsegpt \
          --nsamples 16 \
          --sparsity_ratio 0.5 \
          --sparsity_type unstructured \
          --dataset c4 \
          --eval_zero_shot \
          --save "/mnt/parscratch/users/aca22yn/wanda2/wanda/results/deepseek_7b_sgpt_0.5/" \
          --save_model "/mnt/parscratch/users/aca22yn/wanda2/wanda/pruned/sgpt/deepseek_7b_sgpt_0.5/" \