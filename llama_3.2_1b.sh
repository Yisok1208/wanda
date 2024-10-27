#!/bin/bash
#SBATCH --job-name=llama_3.2_1B_job
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=job_output_llama_3.2_1B_%j.log
#SBATCH --mail-user=yng9@sheffield.ac.uk
#SBATCH --mail-type=ALL

export SLURM_EXPORT_ENV=ALL

# Load any required modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

#Activate conda
conda activate prune_llm
export HUGGINGFACE_TOKEN=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

srun python /mnt/parscratch/users/aca22yn/wanda/main.py \
          --model meta-llama/Llama-3.2-1B-Instruct \
          --prune_method wanda \
          --sparsity_ratio 0.5 \
          --sparsity_type unstructured \
          --save_dir "/mnt/parscratch/users/aca22yn/results/llama_3.2_1B/" \
	  --hf_token 'cat"/mnt/parscratch/users/aca22yn/wanda/token/hf_token.txt

