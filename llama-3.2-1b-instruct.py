from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# 正确的 Hugging Face 模型路径
model_name = "meta-llama/Llama-3.2-1B-Instruct"
filename = "model.safetensors"  # 更新为正确的文件名

# 下载模型的文件
file_path = hf_hub_download(repo_id=model_name, filename=filename, use_auth_token=True)
print(f"模型文件已下载到: {file_path}")

# 使用 transformers 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
print("模型和分词器已下载并加载！")

