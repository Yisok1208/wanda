# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Load and process SST-2 (Sentiment Analysis) dataset
def get_sst2(nsamples, seed, tokenizer):
    print("DEBUG: Loading SST-2 dataset from LOCAL CACHE...")
    cache_path = "/mnt/parscratch/users/aca22yn/cache/datasets/glue/sst2"  # 👈 Your confirmed path
    
    # Debug: Check if cache directory exists
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache directory {cache_path} does not exist!")
        return None
    
    try:
        dataset = load_dataset(
            "glue", "sst2",
            cache_dir=cache_path,  # Directly use the SST-2 subdirectory
            keep_in_memory=True,
            download_mode="force_redownload"
        )
        print(f"DEBUG: Dataset splits: {dataset.keys()}")
        print(f"DEBUG: Train split samples: {len(dataset['train'])}")
    except Exception as e:
        print(f"ERROR: Failed to load SST-2: {str(e)}")
        return None

    # Ensure dataset has enough samples
    if len(dataset["train"]) < nsamples:
        print(f"ERROR: Not enough samples in SST-2. Requested: {nsamples}, Available: {len(dataset['train'])}")
        return None

    # Sample and tokenize
    random.seed(seed)
    sampled_data = random.sample(list(dataset["train"]), nsamples)
    inputs = tokenizer([ex["sentence"] for ex in sampled_data], padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor([ex["label"] for ex in sampled_data])

    return inputs, labels

def get_squad(nsamples, seed, tokenizer):
    print("DEBUG: Loading SQuAD dataset from cache...")

    try:
        dataset = load_dataset("squad", 
                               cache_dir="/mnt/parscratch/users/aca22yn/cache/datasets", 
                               keep_in_memory=True)
        print("DEBUG: SQuAD dataset loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load SQuAD dataset: {e}")
        return None

    # Debug available dataset splits
    print(f"DEBUG: Available dataset splits: {list(dataset.keys())}")

    if "train" not in dataset:
        print("ERROR: No 'train' split found in SQuAD dataset!")
        return None

    # Ensure dataset has enough samples
    if len(dataset["train"]) < nsamples:
        print(f"ERROR: Not enough samples in SQuAD. Requested: {nsamples}, Available: {len(dataset['train'])}")
        return None

    # Sample and tokenize
    random.seed(seed)
    sampled_data = random.sample(list(dataset["train"]), nsamples)
    questions = [ex["question"] for ex in sampled_data]
    contexts = [ex["context"] for ex in sampled_data]
    inputs = tokenizer(questions, contexts, padding=True, truncation=True, return_tensors="pt")

    return inputs, sampled_data  # Keep original data for evaluation

# Function to select dataset loader
def get_loaders(name, nsamples=128, seed=0, tokenizer=None, seqlen=2048):
    # Handle SST-2 and SQuAD
    if "sst2" in name:
        return get_sst2(nsamples, seed, tokenizer)
    elif "squad" in name:
        return get_squad(nsamples, seed, tokenizer)
    # Handle wikitext2 and c4
    elif 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    else:
        print(f"ERROR: Unknown dataset '{name}'")
        return None
    
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
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    cache_dir = '/mnt/parscratch/users/aca22yn/cache/datasets'
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir)
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
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