import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from importlib.metadata import version
from lib.data import get_loaders
from sklearn.metrics import accuracy_score

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="/mnt/parscratch/users/aca22yn/cache/transformers", hf_token=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_auth_token=hf_token,
        force_download=True
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def evaluate_sst2(model, tokenizer, inputs, labels):
    """Evaluates the pruned model on SST-2 (Sentiment Analysis) dataset."""
    print("Evaluating SST-2...")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    predictions = []
    for text in inputs["input_ids"]:
        result = classifier(tokenizer.decode(text))
        predictions.append(1 if result[0]["label"] == "POSITIVE" else 0)

    accuracy = accuracy_score(labels.numpy(), predictions)
    print(f"SST-2 Accuracy: {accuracy:.4f}")
    return accuracy

def evaluate_squad(model, tokenizer, inputs, data):
    """Evaluates the pruned model on SQuAD (Question Answering) dataset."""
    print("Evaluating SQuAD...")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    correct = 0
    total = len(data)

    for i in range(total):
        question = data[i]["question"]
        context = data[i]["context"]
        ground_truth = data[i]["answers"]["text"][0]  # First correct answer

        result = qa_pipeline(question=question, context=context)
        prediction = result["answer"]

        if prediction.lower() == ground_truth.lower():
            correct += 1

    accuracy = correct / total
    print(f"SQuAD Accuracy: {accuracy:.4f}")
    return accuracy

def estimate_snr(t, sparsity):
    # Apply Top-K masking directly
    k = int(t.numel() * (1 - sparsity))  # Number of non-zero elements to retain
    if k == 0:
        t_s = torch.zeros_like(t)
    else:
        t_abs = torch.abs(t)
        topk_values, _ = torch.topk(t_abs.view(-1), k)
        threshold = topk_values[-1]  # Threshold for Top-K
        mask = (t_abs >= threshold).float()
        t_s = mask * t  # Masked tensor

    # Calculate Mean Squared Error (MSE) and Tensor Norm
    mse = torch.mean((t - t_s) ** 2)
    tensor_norm = torch.mean(t ** 2)
    
    # Compute SNR
    if mse.item() > 0.0:
        pruning_snr = 10 * np.log10(tensor_norm.item() / mse.item())
    else:
        pruning_snr = np.Inf
    
    return mse, pruning_snr

def compute_pruning_error(model, original_weights):
    total_error = 0.0
    total_elements = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Retrieve the original and pruned weights
                original_weight = original_weights[name]
                pruned_weight = param.data

                # Compute the L2 difference (pruning error)
                error = torch.sum((original_weight - pruned_weight) ** 2).item()
                total_error += error
                total_elements += param.numel()  # Count the total number of weights
                print(f"Layer: {name} | Pruning Error: {error:.6f}")
    avg_error = total_error / total_elements if total_elements > 0 else 0.0
    return avg_error

def main():
    print("Script started successfully.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="/mnt/parscratch/users/aca22yn/cache/transformers", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, hf_token=os.getenv("HF_TOKEN"))
    print("Model loaded successfully.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
    print("Tokenizer loaded successfully.")

    # Load datasets
    print("DEBUG: Calling get_loaders() for SST-2...")
    sst2_result = get_loaders("sst2", nsamples=128, seed=args.seed, tokenizer=tokenizer)

    # ✅ Check if dataset loading failed
    if sst2_result is None:
        raise ValueError("ERROR: SST-2 dataset could not be loaded! `get_loaders()` returned None.")

    # ✅ Unpack the dataset
    sst2_inputs, sst2_labels = sst2_result
    print(f"DEBUG: SST-2 dataset loaded successfully. Shapes: Inputs={sst2_inputs['input_ids'].shape}, Labels={sst2_labels.shape}")
    
    print("DEBUG: Calling get_loaders() for SQuAD...")
    squad_result = get_loaders("squad", nsamples=128, seed=args.seed, tokenizer=tokenizer)

    # ✅ Check if dataset loading failed
    if squad_result is None:
        raise ValueError("ERROR: SQuAD dataset could not be loaded! `get_loaders()` returned None.")

    # ✅ Unpack the dataset
    squad_inputs, squad_data = squad_result
    print("DEBUG: SQuAD dataset loaded successfully.")

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    original_weights = {name: param.data.clone() for name, param in model.named_parameters() if 'weight' in name}
    
    if args.sparsity_ratio != 0:
        print("Starting pruning with method:", args.prune_method)
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        print("Pruning completed.")

        print("Estimating SNR after pruning...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:  # Focus on weight tensors
                    t = param.data  # Extract the pruned weight tensor
                    sparsity = args.sparsity_ratio
                    mse, pruning_snr = estimate_snr(t, sparsity)
                    print(f"Layer: {name} | MSE: {mse.item():.6f} | SNR: {pruning_snr:.6f}")
                    break  # Only process the first matching weight tensor

        print("Computing pruning error...")
        pruning_error = compute_pruning_error(model, original_weights)
        print(f"Total Pruning Error: {pruning_error:.6f}")

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    print("Starting perplexity evaluation on wikitext.")
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test\tMSE\tSNR\tPruning_Error", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{mse.item():.6f}\t{pruning_snr:.6f}\t{pruning_error:.6f}", file=f, flush=True)

    if args.eval_zero_shot:
        print("Starting zero-shot evaluation.")
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Model and tokenizer saved successfully.")

if __name__ == '__main__':
    main()