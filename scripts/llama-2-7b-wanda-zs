#!/bin/bash
#SBATCH --job-name=llama_2_7B_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=llama_2_7B_hf_wanda_zs_%j.log
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

for s in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  srun python /mnt/parscratch/users/aca22yn/yik/wanda/main.py \
            --model /mnt/parscratch/users/aca22yn/yik/wanda/pruned/sgpt/llama-2-7b-chat-hf_wanda_${s}/ \
            --sparsity_type unstructured \
            --eval_zero_shot \
            --save  /mnt/parscratch/users/aca22yn/yik/wanda/results/llama-2-7b-chat-hf_wanda_${s}_eval/ 
done
