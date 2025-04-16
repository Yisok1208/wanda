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
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = (inp != tokenizer.pad_token_id).long()  # 防止 position_ids 出错
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, attention_mask, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    datasets = load_dataset(
        'allenai/c4',
        'en',
        data_files={
            'train': 'en/c4-train.00000-of-01024.json.gz',
            'validation': 'en/c4-validation.00000-of-00008.json.gz'
        },
        cache_dir="./datasets_cache",
        verification_mode="no_checks",
        download_mode="force_redownload"
    )
    traindata = datasets['train']
    valdata = datasets['validation']

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        # Instead of selecting a single example, concatenate multiple examples until reaching seqlen
        concatenated_text = ""
        while True:
            i = random.randint(0, len(traindata) - 1)
            concatenated_text += " " + traindata[i]['text']
            trainenc = tokenizer(concatenated_text, return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = (inp != tokenizer.pad_token_id).long()  # 如果 tokenizer 没有 pad_token，可设为 0
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, attention_mask, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
