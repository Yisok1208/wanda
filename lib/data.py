# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    import random
    import torch

    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode test dataset for evaluation
    print("Tokenizing wikitext2 test set...")
    test_text = "\n\n".join(testdata['text'][:200])
    testenc = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=256 * seqlen)
    print("Tokenization complete.")

    # Generate training samples (each sample from a single document)
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            encoded = tokenizer(
                traindata[idx]["text"],
                return_tensors="pt",
            )
            if encoded.input_ids.shape[1] >= seqlen:
                break
        start = random.randint(0, encoded.input_ids.shape[1] - seqlen)
        end = start + seqlen
        inp = encoded.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(
                traindata[i]['text'],
                return_tensors='pt',
            )
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(
        ' '.join(valdata[:1100]['text']),
        return_tensors='pt',
        truncation=True,
        max_length=256 * seqlen
    )
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)