#!/bin/bash
#SBATCH --job-name=llama_3.2_1B_job    # Job name
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Total memory limit
#SBATCH --time=12:00:00                # Time limit
#SBATCH --output=job_output_llama_3.2_1B_%j.log   # Standard output and error log
#SBATCH --mail-user=yng9@sheffield.ac.uk        # Email notifications
#SBATCH --mail-type=END            # Send email on end or fail

# Load any required modules or environments
module load Python/3.10.8-GCCcore-12.2.0

# Ensure necessary Python packages are installed in the target directory
pip install --target=/mnt/parscratch/users/aca22yn/python-packages numpy torch transformers datasets accelerate

# Set Hugging Face authentication token (ensure you have access to the required model)
export HUGGINGFACE_TOKEN=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

# Set cache directories for Hugging Face models and datasets
export HF_HOME=/mnt/parscratch/users/aca22yn/.cache/huggingface
export TMPDIR=/mnt/parscratch/users/aca22yn/tmp  # Temporary files directory
export PYTHONPATH=/mnt/parscratch/users/aca22yn/python-packages:$PYTHONPATH

# Ensure necessary directories exist
mkdir -p $HF_HOME /mnt/parscratch/users/aca22yn/results/llama_3.2_1B/unstructured/wanda

# Run your Python script with Llama-3.2-1B-Instruct
python /mnt/parscratch/users/aca22yn/wanda/main.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save /mnt/parscratch/users/aca22yn/results/llama_3.2_1B/unstructured/wanda/

