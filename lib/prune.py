import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    layers = model.model.layers
    count = 0
    total_params = 0
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()
            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
    model.config.use_cache = use_cache
    return float(count) / total_params

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_mask, position_ids = None, None
    for batch in dataloader:
        batch = batch[0].to(device)
        inps.append(batch)
        if attention_mask is None:
            attention_mask = torch.ones_like(batch)
            position_ids = torch.arange(batch.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        if len(inps) >= 128:
            break
    model.config.use_cache = use_cache
    return torch.stack(inps), attention_mask, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = W_metric <= thres
    cur_sparsity = W_mask.sum().item() / W_mask.numel()
    return W_mask, cur_sparsity

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting prune_sparsegpt...', flush=True)
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print('Calibration dataloader loaded.', flush=True)
    use_cache = model.config.use_cache
    layers = model.model.layers
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
    dtype = next(iter(model.parameters())).dtype
    inps, attention_mask, position_ids = prepare_calibration_input(model, dataloader, dev)
    outs = torch.zeros_like(inps, dtype=dtype)
    print('Ready.')

    for i, layer in enumerate(layers):
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        subset = find_layers(layer)
        gpts = {name: SparseGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]

        for j in range(args.nsamples):
            outs[j] = model(
                input_ids=inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                use_cache=False
            ).logits[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = model(
                input_ids=inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                use_cache=False
            ).logits[0]

        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    print("loading calibration data")
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    inps, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, attention_mask, position_ids = inps.to(dev), attention_mask.to(dev), position_ids.to(dev)
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        for j in range(args.nsamples):
            _ = model(
                input_ids=inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                use_cache=False
            ).logits[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = torch.zeros_like(W_metric).bool()
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while abs(cur_sparsity - args.sparsity_ratio) > 0.001 and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        alpha_new = (alpha + alpha_hist[0]) / 2.0 if cur_sparsity > args.sparsity_ratio else (alpha + alpha_hist[1]) / 2.0
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_hist[1] = alpha
                        else:
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0

        torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
