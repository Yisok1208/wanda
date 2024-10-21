# prune_script.py
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置缓存目录
cache_dir = '/mnt/parscratch/users/aca22yn/huggingface_cache'

# 加载数据集并指定缓存目录
traindata = load_dataset('c4', 'en', split='train', cache_dir=cache_dir)
valdata = load_dataset('c4', 'en', split='validation', cache_dir=cache_dir)

# 加载模型并指定缓存目录
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', cache_dir=cache_dir)

# 剪枝逻辑在此添加...

