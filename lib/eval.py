# Import necessary modules
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from evaluate import load

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch

def evaluate_triviaqa(model, tokenizer, max_samples=500):
    """
    Evaluate pruned LLM on TriviaQA dataset using Exact Match (EM) and F1 metrics.
    
    Args:
        model: Pruned language model
        tokenizer: Model's tokenizer
        max_samples: Maximum samples to evaluate (for faster testing)
    
    Returns:
        exact_match (float): EM score
        f1_score (float): F1 score
    """
    # Load TriviaQA dataset (unfiltered split)
    dataset = load_dataset("trivia_qa", "unfiltered")["test"]
    dataset = dataset.select(range(max_samples))  # Limit samples for quick evaluation
    
    # Load evaluation metric (SQuAD-style EM/F1)
    qa_metric = load("squad")
    
    predictions = []
    references = []
    
    for example in dataset:
        # Construct prompt with explicit instruction
        prompt = (
            f"Question: {example['question']}\n"
            "Answer in 1-5 words:"  # Force concise answers
        )
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=20,  # Restrict answer length
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and clean answer
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = full_response.split("Answer in 1-5 words:")[-1].strip()
        
        # Store predictions and references
        predictions.append({"id": example["id"], "prediction_text": answer})
        references.append({
            "id": example["id"], 
            "answers": {
                "text": [example["answer"]["value"]],  # Required format for SQuAD metric
                "answer_start": [0]
            }
        })
    
    # Compute metrics
    results = qa_metric.compute(
        predictions=predictions,
        references=references
    )
    
    return results["exact_match"], results["f1"]

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )
    print(f"Loaded {len(testloader)} samples in testloader.")
    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        print(f"Processing batch {i // bs + 1}/{nsamples // bs + 1}")
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","triviaqa","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator
    from .your_custom_module import evaluate_triviaqa  # 导入自定义评估函数

    # 分离内置任务和自定义任务
    builtin_tasks = [t for t in task_list if t in tasks.ALL_TASKS]
    custom_tasks = [t for t in task_list if t not in tasks.ALL_TASKS]

    results = {}

    # 评估内置任务
    if builtin_tasks:
        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        task_names = pattern_match(builtin_tasks, tasks.ALL_TASKS)
        model_args = f"pretrained={model_name},cache_dir=./llm_weights"
        
        if "70b" in model_name or "65b" in model_name:
            limit = 2000
        else:
            limit = None
            
        if use_accelerate:
            model_args += ",use_accelerate=True"

        builtin_results = evaluator.simple_evaluate(
            model="hf-causal-experimental",
            model_args=model_args,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=None,
            device=None,
            no_cache=True,
            limit=limit,
            description_dict={},
            decontamination_ngrams_path=None,
            check_integrity=False,
            pretrained_model=model,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens
        )
        results.update(builtin_results["results"])

    # 评估自定义任务
    if "triviaqa" in custom_tasks:
        em_score, f1_score = evaluate_triviaqa(model, tokenizer)
        results["triviaqa"] = {
            "exact_match": em_score,
            "f1": f1_score
        }

    return results