# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

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
    testenc = testenc.input_ids
    max_seq_len = 512  # 显存友好

    total_tokens = testenc.shape[1]
    nsamples = total_tokens // max_seq_len
    nlls = []

    print(f"nsamples {nsamples}")
    for i in range(nsamples):
        start = i * max_seq_len
        end = start + max_seq_len
        inputs = testenc[:, start:end].to(device)

        if inputs.shape[1] < 2:
            continue  # skip too-short input

        try:
            outputs = model(inputs)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on sample {i}, skipping")
            torch.cuda.empty_cache()
            continue

        lm_logits = outputs.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        neg_log_likelihood = loss.float() * (inputs.shape[1] - 1)
        nlls.append(neg_log_likelihood)

    if len(nlls) == 0:
        raise ValueError("No samples could be processed without OOM.")

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (max_seq_len - 1)))
    torch.cuda.empty_cache()
    return ppl.item()

def eval_zero_shot(
    model_name,
    model,
    tokenizer,
    task_list = ["boolq","rte","hellaswag","winogrande",
                 "arc_challenge","arc_easy","openbookqa"],
    num_fewshot = 0,
    use_accelerate = False,
    add_special_tokens = False,
):
    from lm_eval import evaluator
    task_names = task_list
    
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    if use_accelerate:
        model_args += ",use_accelerate=True"
    limit = 2000 if any(x in model_name for x in ("70b", "65b")) else None

    results = evaluator.simple_evaluate(
        model              = "hf",
        model_args         = model_args,
        tasks              = task_names,
        num_fewshot        = num_fewshot,
        no_cache           = True,
        limit              = limit,
        check_integrity    = False,
        tokenizer          = tokenizer,
        add_special_tokens = add_special_tokens,
    )
    return results
