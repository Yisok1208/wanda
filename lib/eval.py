# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 
from lm_eval.tasks import get_task_dict
from collections import defaultdict
import fnmatch


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
def eval_ppl_wikitext(model, testenc, tokenizer, bs=1, device=None):
    if not hasattr(testenc, "input_ids") or testenc.input_ids is None:
        print("ERROR: testenc.input_ids empty or not exist")
        return None
    
    print(f"testenc.input_ids.shape: {testenc.input_ids.shape}")

    max_length = getattr(tokenizer, "model_max_length", 16384)
    testenc_ids = testenc.input_ids[:, :max_length].clone().detach()
    print(f"Limiting testenc to max_length={max_length}")

    nsamples = testenc_ids.numel() // model.seqlen
    nlls = []
    print(f"nsamples: {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"Processing sample {i}")

        j = min(i + bs, nsamples)
        inputs = testenc_ids[:, (i * model.seqlen):(j * model.seqlen)].to(device)

        print(f"Batch {i//bs}: inputs.shape before reshape: {inputs.shape}")
        inputs = inputs.reshape(j - i, model.seqlen)
        print(f"Batch {i//bs}: inputs.shape after reshape: {inputs.shape}")

        try:
            lm_logits = model(inputs).logits
            print(f"Batch {i//bs}: Forward pass successful! logits shape: {lm_logits.shape}")
        except Exception as e:
            print(f"Batch {i//bs}: Error in model forward pass: {e}")
            return None

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, tokenizer, bs=min(bs, nsamples), device=None):

    if hasattr(testenc, 'input_ids'):
        print(f"testenc.input_ids.shape: {testenc.input_ids.shape}")
    else:
        print("WARNING: testenc does not have input_ids!")
        return None
    
    print(f"testenc Type: {type(testenc)}")
    print(f"testenc.input_ids.shape: {testenc.input_ids.shape}" if hasattr(testenc, "input_ids") else "testenc.input_ids Not Exist")

    max_length = getattr(tokenizer, "model_max_length", 16384)
    testenc.input_ids = testenc.input_ids[:, :max_length]
    print(f"Limiting testenc to max_length={max_length}")

    nsamples = testenc.input_ids.numel() // model.seqlen
    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)

        # 调试：检查 inputs 形状
        print(f"Batch {i//bs}: inputs.shape before reshape: {inputs.shape}")
        inputs = inputs.reshape(j-i, model.seqlen)
        print(f"Batch {i//bs}: inputs.shape after reshape: {inputs.shape}")

        try:
            lm_logits = model(inputs).logits
            print(f"Batch {i//bs}: Forward pass successful! logits shape: {lm_logits.shape}")
        except Exception as e:
            print(f"Batch {i//bs}: Error in model forward pass: {e}")
            return None  # 避免 CUDA 崩溃

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # # 避免 CUDA 崩溃，先注释掉 empty_cache()
    # torch.cuda.empty_cache()

    return ppl.item()

def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, list(get_task_dict(task_list).keys()))
    model_args = f"pretrained={model_name}"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf",
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
    )

    return results 